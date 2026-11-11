"""Stage 2: Multimodal ESM-Augmented Fusion ResNet for Pathogenicity Prediction.

Architecture:
  - Tabular branch: Linear -> LayerNorm -> GELU (continuous + binary features)
  - Categorical branch: variant_class embedding (12), syndrome embedding (15)
    (gene embedding is OFF by default to prevent gene memorization; flag --use_gene_embedding)
  - ESM branch: [esm_llr, esm_available, Linear(480 -> 64)] -> MLP(64)
  - Fusion: Concat -> Linear(..., 256) -> 2 Residual Blocks (LayerNorm, GELU, dropout 0.2) -> Head -> Logit

Training:
  - Weighted BCEWithLogitsLoss (pos_weight = n_neg / n_pos on train only)
  - AdamW (lr=1e-3, weight_decay=1e-4) + CosineAnnealingLR
  - Early stopping on validation AUPRC
  - Multi-seed averaging (default 5 seeds)
  - Temperature scaling calibration on validation set; report ECE before/after

Checkpoint stores:
  model_state_dict, scaler, continuous_cols, binary_cols, panel_json,
  esm_model_name, calibration_temperature, ece_before, ece_after, metrics
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from genomedl.features.esm_features import load_or_reconstruct_embeddings
from genomedl.paths import (
    DEFAULT_CONFIG,
    ESM_CACHE_PARQUET,
    ESM_EMBEDDINGS_PT,
    FEATURED_SNV_CSV,
    FUSION_META_JSON,
    FUSION_MODEL_PTH,
    PANEL_JSON,
    load_config,
    resolve_project_path,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_fusion")

FEATURED_CSV = FEATURED_SNV_CSV
EMBEDDINGS_PT = ESM_EMBEDDINGS_PT

CONTINUOUS_COLS = [
    "rel_protein_pos",
    "grantham_score",
    "delta_mw",
    "delta_charge",
    "delta_logp",
    "delta_tpsa",
    "tanimoto_chem_dist",
]

BINARY_COLS = [
    "on_primary_transcript",
    "has_protein_change",
    "is_transition",
    "ref_A", "alt_A", "ref_C", "alt_C", "ref_G", "alt_G", "ref_T", "alt_T",
    "cs_ref_A", "cs_alt_A", "cs_ref_C", "cs_alt_C", "cs_ref_G", "cs_alt_G", "cs_ref_T", "cs_alt_T",
]


class GenomicDataset(Dataset):
    """Dataset for Tabular + Categorical + ESM multimodal features."""

    def __init__(
        self,
        tabular_feats: np.ndarray,
        variant_class_ids: np.ndarray,
        syndrome_ids: np.ndarray,
        gene_ids: Optional[np.ndarray],
        esm_llrs: np.ndarray,
        esm_availables: np.ndarray,
        esm_embeddings: torch.Tensor,
        labels: Optional[np.ndarray] = None,
    ):
        self.tabular = torch.tensor(tabular_feats, dtype=torch.float32)
        self.vclass = torch.tensor(variant_class_ids, dtype=torch.long)
        self.syndrome = torch.tensor(syndrome_ids, dtype=torch.long)
        self.gene = torch.tensor(gene_ids, dtype=torch.long) if gene_ids is not None else None
        self.esm_llr = torch.tensor(esm_llrs, dtype=torch.float32).unsqueeze(-1)
        self.esm_avail = torch.tensor(esm_availables, dtype=torch.float32).unsqueeze(-1)
        self.esm_emb = esm_embeddings  # float16 or float32 tensor
        self.labels = torch.tensor(labels, dtype=torch.float32).unsqueeze(-1) if labels is not None else None

    def __len__(self) -> int:
        return len(self.tabular)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = {
            "tabular": self.tabular[idx],
            "vclass": self.vclass[idx],
            "syndrome": self.syndrome[idx],
            "esm_llr": self.esm_llr[idx],
            "esm_avail": self.esm_avail[idx],
            "esm_emb": self.esm_emb[idx].to(torch.float32),
        }
        if self.gene is not None:
            item["gene"] = self.gene[idx]
        if self.labels is not None:
            item["label"] = self.labels[idx]
        return item


class ResBlock(nn.Module):
    """Residual block with LayerNorm, GELU, and Dropout."""

    def __init__(self, dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FusionResNet(nn.Module):
    """Multimodal Fusion ResNet with Tabular, Categorical, and ESM branches."""

    def __init__(
        self,
        tabular_dim: int,
        num_variant_classes: int = 12,
        num_syndromes: int = 15,
        num_genes: Optional[int] = None,
        use_gene_embedding: bool = False,
        esm_emb_dim: int = 480,
        esm_proj_dim: int = 64,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.use_gene_embedding = use_gene_embedding

        # 1. Tabular branch
        self.tabular_branch = nn.Sequential(
            nn.Linear(tabular_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
        )

        # 2. Categorical embeddings
        self.vclass_emb = nn.Embedding(num_variant_classes, 8)
        self.syndrome_emb = nn.Embedding(num_syndromes, 8)
        cat_dim = 16
        if self.use_gene_embedding and num_genes is not None:
            self.gene_emb = nn.Embedding(num_genes, 16)
            cat_dim += 16
        else:
            self.gene_emb = None

        # 3. ESM branch
        self.esm_proj = nn.Linear(esm_emb_dim, esm_proj_dim)
        # Input to ESM MLP: esm_llr (1) + esm_avail (1) + esm_proj (64) = 66
        self.esm_mlp = nn.Sequential(
            nn.Linear(1 + 1 + esm_proj_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(64, 64),
        )

        # 4. Fusion layer
        total_in_dim = 64 + cat_dim + 64
        self.fusion_in = nn.Sequential(
            nn.Linear(total_in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # 5. Residual blocks
        self.res1 = ResBlock(hidden_dim, dropout=dropout)
        self.res2 = ResBlock(hidden_dim, dropout=dropout)

        # 6. Prediction Head -> one logit
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        tabular: torch.Tensor,
        vclass: torch.Tensor,
        syndrome: torch.Tensor,
        esm_llr: torch.Tensor,
        esm_avail: torch.Tensor,
        esm_emb: torch.Tensor,
        gene: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Tabular
        h_tab = self.tabular_branch(tabular)

        # Categoricals
        h_vclass = self.vclass_emb(vclass)
        h_synd = self.syndrome_emb(syndrome)
        cat_feats = [h_vclass, h_synd]
        if self.use_gene_embedding and self.gene_emb is not None and gene is not None:
            cat_feats.append(self.gene_emb(gene))
        h_cat = torch.cat(cat_feats, dim=-1)

        # ESM
        proj_esm = self.esm_proj(esm_emb)
        esm_concat = torch.cat([esm_llr, esm_avail, proj_esm], dim=-1)
        h_esm = self.esm_mlp(esm_concat)

        # Fusion
        fused = torch.cat([h_tab, h_cat, h_esm], dim=-1)
        x = self.fusion_in(fused)
        x = self.res1(x)
        x = self.res2(x)
        logits = self.head(x)
        return logits


class TemperatureScaler(nn.Module):
    """Calibrates model logits with Platt scaling: scale * logit + bias.
    
    Under class imbalance with pos_weight > 1, raw uncalibrated logits are shifted
    by approximately +log(pos_weight). Platt scaling fits both scale (1/T) and bias (offset),
    centering calibrated probabilities to match empirical prevalence and minimizing
    Expected Calibration Error (ECE).
    """

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1) * 0.8)
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return self.scale * logits + self.bias

    @property
    def temperature(self) -> float:
        s = self.scale.item()
        return 1.0 / s if abs(s) > 1e-6 else 1.0

    def fit(self, logits: torch.Tensor, labels: torch.Tensor, lr: float = 0.05, max_iter: int = 200) -> Tuple[float, float]:
        """Fit Platt scaling parameters (scale, bias) on validation logits and labels via BCE."""
        self.train()
        optimizer = torch.optim.LBFGS([self.scale, self.bias], lr=lr, max_iter=max_iter)
        criterion = nn.BCEWithLogitsLoss()

        def eval_loss():
            optimizer.zero_grad()
            loss = criterion(self.forward(logits), labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)
        self.eval()
        scale_val = float(self.scale.item())
        bias_val = float(self.bias.item())
        t_val = self.temperature
        logger.info(f"Fitted Platt calibration: scale = {scale_val:.4f} (equiv T = {t_val:.4f}), bias = {bias_val:.4f}")
        return scale_val, bias_val


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> float:
    """Expected Calibration Error (ECE)."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        in_bin = (probs >= bin_lower) & (probs < bin_upper) if i < n_bins - 1 else (probs >= bin_lower) & (probs <= bin_upper)
        prop_in_bin = np.mean(in_bin)
        if prop_in_bin > 0:
            accuracy_in_bin = np.mean(labels[in_bin])
            avg_confidence_in_bin = np.mean(probs[in_bin])
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return float(ece)


def train_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: str) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        tabular = batch["tabular"].to(device)
        vclass = batch["vclass"].to(device)
        syndrome = batch["syndrome"].to(device)
        esm_llr = batch["esm_llr"].to(device)
        esm_avail = batch["esm_avail"].to(device)
        esm_emb = batch["esm_emb"].to(device)
        gene = batch["gene"].to(device) if "gene" in batch else None
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(tabular, vclass, syndrome, esm_llr, esm_avail, esm_emb, gene)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for batch in loader:
            tabular = batch["tabular"].to(device)
            vclass = batch["vclass"].to(device)
            syndrome = batch["syndrome"].to(device)
            esm_llr = batch["esm_llr"].to(device)
            esm_avail = batch["esm_avail"].to(device)
            esm_emb = batch["esm_emb"].to(device)
            gene = batch["gene"].to(device) if "gene" in batch else None
            labels = batch["label"].cpu().numpy()

            logits = model(tabular, vclass, syndrome, esm_llr, esm_avail, esm_emb, gene)
            all_logits.append(logits.cpu().numpy())
            all_labels.append(labels)

    logits_arr = np.concatenate(all_logits, axis=0).flatten()
    labels_arr = np.concatenate(all_labels, axis=0).flatten()
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))
    return logits_arr, probs_arr, labels_arr


def train_single_seed(
    train_ds: GenomicDataset,
    val_ds: GenomicDataset,
    test_ds: GenomicDataset,
    tabular_dim: int,
    num_genes: Optional[int],
    use_gene_embedding: bool,
    pos_weight: float,
    seed: int,
    device: str,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    dropout: float = 0.2,
    hidden_dim: int = 256,
    esm_proj_dim: int = 64,
    grad_clip_norm: float = 1.0,
) -> Tuple[FusionResNet, float, Dict]:
    """Train one model instance with early stopping on validation AUPRC."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=(device == "cuda"))
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    model = FusionResNet(
        tabular_dim=tabular_dim,
        num_variant_classes=12,
        num_syndromes=15,
        num_genes=num_genes,
        use_gene_embedding=use_gene_embedding,
        esm_emb_dim=480,
        esm_proj_dim=esm_proj_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    ).to(device)

    pos_w_tensor = torch.tensor([pos_weight], device=device, dtype=torch.float32)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w_tensor)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_auprc = 0.0
    best_weights = None
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, criterion, optimizer, device)
        scheduler.step()

        _, val_probs, val_labels = evaluate(model, val_loader, device)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_auprc = average_precision_score(val_labels, val_probs)

        if val_auprc > best_val_auprc:
            best_val_auprc = val_auprc
            best_weights = copy.deepcopy(model.state_dict())
            patience_counter = 0
            best_epoch = epoch
        else:
            patience_counter += 1

        if epoch % 5 == 0 or epoch == epochs or patience_counter == 0:
            logger.info(f"Seed {seed} Epoch {epoch:2d}/{epochs} | Loss: {loss:.4f} | Val AUC: {val_auc:.4f} | Val AUPRC: {val_auprc:.4f} {'*' if patience_counter == 0 else ''}")

        if patience_counter >= patience:
            logger.info(f"Seed {seed} early stopping triggered at epoch {epoch}. Best at epoch {best_epoch} (Val AUPRC: {best_val_auprc:.4f}).")
            break

    model.load_state_dict(best_weights)

    # Test evaluation
    _, test_probs, test_labels = evaluate(model, test_loader, device)
    test_auc = roc_auc_score(test_labels, test_probs)
    test_auprc = average_precision_score(test_labels, test_probs)

    metrics = {
        "best_epoch": best_epoch,
        "best_val_auprc": best_val_auprc,
        "test_auc": test_auc,
        "test_auprc": test_auprc,
    }
    logger.info(f"Seed {seed} final Test AUC: {test_auc:.4f} | Test AUPRC: {test_auprc:.4f}")
    return model, best_val_auprc, metrics


def run_training_pipeline(
    csv_path: Path = FEATURED_CSV,
    embeddings_path: Path = EMBEDDINGS_PT,
    esm_cache_path: Path = ESM_CACHE_PARQUET,
    panel_json_path: Path = PANEL_JSON,
    output_model: Path = FUSION_MODEL_PTH,
    output_meta: Path = FUSION_META_JSON,
    use_gene_embedding: bool = False,
    num_seeds: int = 5,
    seeds: Optional[List[int]] = None,
    epochs: int = 25,
    batch_size: int = 256,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 7,
    dropout: float = 0.2,
    hidden_dim: int = 256,
    esm_proj_dim: int = 64,
    device: Optional[str] = None,
):
    """Main training, multi-seed ensembling, temperature calibration, and checkpointing."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Loading data for Stage 2 Fusion ResNet (use_gene_embedding={use_gene_embedding}, device={device})...")
    df = pd.read_csv(csv_path, low_memory=False)
    panel_dict = json.load(open(panel_json_path, "r", encoding="utf-8"))

    # Load embeddings tensor
    logger.info(f"Loading embeddings tensor from {embeddings_path}...")
    esm_embeddings = load_or_reconstruct_embeddings(
        featured_csv=csv_path,
        cache_parquet=esm_cache_path,
        embeddings_pt=embeddings_path,
    )

    # Prepare tabular features
    cont_feats = df[CONTINUOUS_COLS].fillna(0.0).values.astype(np.float32)
    bin_feats = df[BINARY_COLS].fillna(0.0).values.astype(np.float32)

    # Prepare categorical features
    vclass_ids = df["variant_class_id"].values.astype(np.int64)
    syndrome_ids = df["syndrome_id"].values.astype(np.int64)
    gene_ids = df["gene_id"].values.astype(np.int64) if use_gene_embedding else None

    # Prepare ESM features
    esm_llrs = df["esm_llr"].fillna(0.0).values.astype(np.float32)
    esm_avails = df["esm_available"].fillna(0).values.astype(np.float32)

    labels = df["label"].values.astype(np.float32)

    # Stratified 70/15/15 split
    indices = np.arange(len(df))
    train_idx, val_test_idx = train_test_split(indices, test_size=0.30, stratify=labels, random_state=42)
    val_idx, test_idx = train_test_split(val_test_idx, test_size=0.50, stratify=labels[val_test_idx], random_state=42)

    logger.info(f"Dataset split: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}")

    # Standard scale continuous features fitted on train only
    scaler = StandardScaler()
    cont_train = scaler.fit_transform(cont_feats[train_idx])
    cont_val = scaler.transform(cont_feats[val_idx])
    cont_test = scaler.transform(cont_feats[test_idx])

    tab_train = np.hstack([cont_train, bin_feats[train_idx]])
    tab_val = np.hstack([cont_val, bin_feats[val_idx]])
    tab_test = np.hstack([cont_test, bin_feats[test_idx]])
    tabular_dim = tab_train.shape[1]

    # Weighted BCE pos_weight on train set only
    n_pos = float(np.sum(labels[train_idx] == 1))
    n_neg = float(np.sum(labels[train_idx] == 0))
    pos_weight = n_neg / n_pos
    logger.info(f"Class imbalance on train set: Neg={int(n_neg)}, Pos={int(n_pos)}, pos_weight={pos_weight:.2f}")

    train_ds = GenomicDataset(
        tab_train, vclass_ids[train_idx], syndrome_ids[train_idx],
        gene_ids[train_idx] if gene_ids is not None else None,
        esm_llrs[train_idx], esm_avails[train_idx], esm_embeddings[train_idx], labels[train_idx]
    )
    val_ds = GenomicDataset(
        tab_val, vclass_ids[val_idx], syndrome_ids[val_idx],
        gene_ids[val_idx] if gene_ids is not None else None,
        esm_llrs[val_idx], esm_avails[val_idx], esm_embeddings[val_idx], labels[val_idx]
    )
    test_ds = GenomicDataset(
        tab_test, vclass_ids[test_idx], syndrome_ids[test_idx],
        gene_ids[test_idx] if gene_ids is not None else None,
        esm_llrs[test_idx], esm_avails[test_idx], esm_embeddings[test_idx], labels[test_idx]
    )

    # Train ensemble seeds
    seed_pool = seeds if seeds else [42, 101, 2024, 7, 999]
    active_seeds = seed_pool[:num_seeds]
    trained_models = []
    seed_metrics = []

    for seed in active_seeds:
        logger.info(f"\n--- Training Seed {seed} ---")
        m, val_auprc, met = train_single_seed(
            train_ds, val_ds, test_ds,
            tabular_dim=tabular_dim,
            num_genes=len(panel_dict) if use_gene_embedding else None,
            use_gene_embedding=use_gene_embedding,
            pos_weight=pos_weight,
            seed=seed,
            device=device,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            patience=patience,
            dropout=dropout,
            hidden_dim=hidden_dim,
            esm_proj_dim=esm_proj_dim,
        )
        trained_models.append(m)
        seed_metrics.append(met)

    # Ensemble evaluation (average logits across seeds)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size * 2, shuffle=False)

    def ensemble_predict(models: List[nn.Module], loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        all_logits = []
        for m in models:
            logits, _, lbls = evaluate(m, loader, device)
            all_logits.append(logits)
        avg_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
        probs = 1.0 / (1.0 + np.exp(-avg_logits))
        return avg_logits, probs, lbls

    val_logits, val_probs, val_labels = ensemble_predict(trained_models, val_loader)
    test_logits, test_probs, test_labels = ensemble_predict(trained_models, test_loader)

    # Calibration via Platt Scaling
    logger.info("\n--- Platt Scaling Calibration ---")
    scaler_temp = TemperatureScaler().to(device)
    val_logits_t = torch.tensor(val_logits, device=device, dtype=torch.float32).unsqueeze(-1)
    val_labels_t = torch.tensor(val_labels, device=device, dtype=torch.float32).unsqueeze(-1)
    fitted_scale, fitted_bias = scaler_temp.fit(val_logits_t, val_labels_t)
    fitted_temp = scaler_temp.temperature

    # Calibrate test set
    test_logits_t = torch.tensor(test_logits, device=device, dtype=torch.float32).unsqueeze(-1)
    calibrated_test_logits = scaler_temp(test_logits_t).detach().cpu().numpy().flatten()
    calibrated_test_probs = 1.0 / (1.0 + np.exp(-calibrated_test_logits))

    # Calculate ECE before and after
    ece_before = compute_ece(test_probs, test_labels, n_bins=10)
    ece_after = compute_ece(calibrated_test_probs, test_labels, n_bins=10)

    test_auc = roc_auc_score(test_labels, calibrated_test_probs)
    test_auprc = average_precision_score(test_labels, calibrated_test_probs)

    logger.info(f"Uncalibrated Test ECE: {ece_before:.4f}")
    logger.info(f"Calibrated Test ECE:   {ece_after:.4f} (improved by {((ece_before - ece_after)/ece_before)*100:.1f}%)")
    logger.info(f"Final Calibrated Test AUC:   {test_auc:.4f}")
    logger.info(f"Final Calibrated Test AUPRC: {test_auprc:.4f}")

    # Per-class AUC / AUPRC on test set (canonical order from build_features.py)
    test_vclass = vclass_ids[test_idx]
    class_metrics = {}
    class_names = [
        "synonymous", "missense", "nonsense", "start_lost", "stop_lost",
        "splice_canonical", "splice_region", "intronic", "utr5", "utr3",
        "other_noncoding", "coding_no_protein_annot"
    ]
    for cid, cname in enumerate(class_names):
        c_mask = test_vclass == cid
        c_pos = int(np.sum(test_labels[c_mask] == 1))
        c_neg = int(np.sum(test_labels[c_mask] == 0))
        if c_pos > 0 and c_neg > 0:
            c_auc = roc_auc_score(test_labels[c_mask], calibrated_test_probs[c_mask])
            c_auprc = average_precision_score(test_labels[c_mask], calibrated_test_probs[c_mask])
            class_metrics[cname] = {
                "n_samples": int(np.sum(c_mask)),
                "n_pos": c_pos,
                "n_neg": c_neg,
                "auc": float(c_auc),
                "auprc": float(c_auprc),
            }
            logger.info(f"  Class {cname:20s}: N={int(np.sum(c_mask)):5d} (pos={c_pos:4d}, neg={c_neg:5d}) | AUC={c_auc:.4f} | AUPRC={c_auprc:.4f}")

    # Save checkpoint
    output_model.parent.mkdir(parents=True, exist_ok=True)
    output_meta.parent.mkdir(parents=True, exist_ok=True)
    # Save best model state_dict (seed 42 / primary)
    checkpoint = {
        "model_state_dict": trained_models[0].state_dict(),
        "ensemble_state_dicts": [m.state_dict() for m in trained_models],
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
        "continuous_cols": CONTINUOUS_COLS,
        "binary_cols": BINARY_COLS,
        "variant_classes": class_names,
        "syndromes": list(set(df["syndrome_group"].unique())),
        "panel_json": panel_dict,
        "esm_model_name": "facebook/esm2_t12_35M_UR50D",
        "calibration_temperature": float(fitted_temp),
        "calibration_scale": float(fitted_scale),
        "calibration_bias": float(fitted_bias),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "test_auc": float(test_auc),
        "test_auprc": float(test_auprc),
        "pos_weight": float(pos_weight),
        "use_gene_embedding": use_gene_embedding,
        "split_protocol": "70/15/15 stratified random split by label",
    }
    torch.save(checkpoint, output_model)
    logger.info(f"Saved checkpoint to {output_model}")

    metadata = {
        "test_auc": float(test_auc),
        "test_auprc": float(test_auprc),
        "ece_before": float(ece_before),
        "ece_after": float(ece_after),
        "temperature": float(fitted_temp),
        "calibration_scale": float(fitted_scale),
        "calibration_bias": float(fitted_bias),
        "seed_metrics": seed_metrics,
        "class_metrics": class_metrics,
        "use_gene_embedding": use_gene_embedding,
        "num_seeds": num_seeds,
    }
    with open(output_meta, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {output_meta}")

    return checkpoint, metadata


def parse_args():
    parser = argparse.ArgumentParser(description="Train Fusion ResNet for GenomeDL v3")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--csv_path", type=Path, default=None)
    parser.add_argument("--embeddings_path", type=Path, default=None)
    parser.add_argument("--output_model", type=Path, default=None)
    parser.add_argument("--output_meta", type=Path, default=None)
    parser.add_argument("--use_gene_embedding", action="store_true", default=None, help="Enable gene embedding (default: OFF to prevent memorization)")
    parser.add_argument("--num_seeds", type=int, default=None, help="Number of seeds to train and average")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    paths_cfg = cfg.get("paths", {})
    model_cfg = cfg.get("model", {})
    train_cfg = cfg.get("training", {})

    csv_path = args.csv_path or (resolve_project_path(paths_cfg["featured_snv_csv"]) if "featured_snv_csv" in paths_cfg else FEATURED_CSV)
    embeddings_path = args.embeddings_path or (resolve_project_path(paths_cfg["esm_embeddings_pt"]) if "esm_embeddings_pt" in paths_cfg else EMBEDDINGS_PT)
    esm_cache_path = resolve_project_path(paths_cfg["esm_cache_parquet"]) if "esm_cache_parquet" in paths_cfg else ESM_CACHE_PARQUET
    panel_json_path = resolve_project_path(paths_cfg["panel_json"]) if "panel_json" in paths_cfg else PANEL_JSON
    output_model = args.output_model or (resolve_project_path(paths_cfg["model_checkpoint"]) if "model_checkpoint" in paths_cfg else FUSION_MODEL_PTH)
    output_meta = args.output_meta or (resolve_project_path(paths_cfg["model_metadata"]) if "model_metadata" in paths_cfg else FUSION_META_JSON)

    use_gene_emb = bool(args.use_gene_embedding) if args.use_gene_embedding is not None else bool(model_cfg.get("use_gene_embedding", False))
    num_seeds = args.num_seeds if args.num_seeds is not None else int(train_cfg.get("num_seeds", 5))
    epochs = args.epochs if args.epochs is not None else int(train_cfg.get("epochs", 25))
    batch_size = args.batch_size if args.batch_size is not None else int(train_cfg.get("batch_size", 256))

    run_training_pipeline(
        csv_path=csv_path,
        embeddings_path=embeddings_path,
        esm_cache_path=esm_cache_path,
        panel_json_path=panel_json_path,
        output_model=output_model,
        output_meta=output_meta,
        use_gene_embedding=use_gene_emb,
        num_seeds=num_seeds,
        seeds=train_cfg.get("seeds"),
        epochs=epochs,
        batch_size=batch_size,
        lr=float(train_cfg.get("lr", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
        patience=int(train_cfg.get("patience", 7)),
        dropout=float(model_cfg.get("dropout", 0.2)),
        hidden_dim=int(model_cfg.get("hidden_dim", 256)),
        esm_proj_dim=int(model_cfg.get("esm_proj_dim", 64)),
        device=args.device,
    )


if __name__ == "__main__":
    main()
