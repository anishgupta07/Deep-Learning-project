"""
train_resnet_v2.py — GenomeDL v2 Training Script
==================================================
Trains a tabular ResNet on v2_clinvar_featured_snv.csv.

Key v1 → v2 changes:
  - 130 genes (was 93), 15 syndrome groups (was 23)
  - variant_class_id embedding replaces is_missense/nonsense/synonymous flags
  - Full 190-pair Grantham matrix already in the CSV
  - Stratified 70/15/15 split
  - Reports overall AUC + per-class AUC (missense & splice_region highlighted)
  - Saves: models/genome_dl_resnet_v2.pth (checkpoint with scaler + mappings)

Usage:
    python train_resnet_v2.py
    python train_resnet_v2.py --epochs 30 --batch_size 512 --seed 42
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# ─────────────────────────── CLI ────────────────────────────
parser = argparse.ArgumentParser(description="GenomeDL v2 — ResNet training")
parser.add_argument("--data",       default="data/v2_clinvar_featured_snv.csv")
parser.add_argument("--outdir",     default="models")
parser.add_argument("--epochs",     type=int, default=25)
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--seed",       type=int, default=42)
parser.add_argument("--lr",         type=float, default=1e-3)
parser.add_argument("--wd",         type=float, default=1e-4)
args = parser.parse_args()

SEED = args.seed
torch.manual_seed(SEED)
np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ─────────────────────── Feature definitions ────────────────
# Columns to standardise (zero chemistry columns → scaler handles gracefully)
SCALE_COLS = [
    "grantham_score", "delta_mw", "delta_charge", "delta_logp",
    "delta_tpsa", "tanimoto_chem_dist", "rel_protein_pos",
]

# Remaining continuous columns (already 0/1, no scaling needed)
BINARY_COLS = [
    "has_protein_change", "is_transition", "on_primary_transcript",
    "ref_A", "ref_C", "ref_G", "ref_T",
    "alt_A", "alt_C", "alt_G", "alt_T",
]

# Categorical IDs → embeddings
CAT_COLS = ["gene_id", "syndrome_id", "variant_class_id"]

# Target
LABEL_COL = "label"

# Variant class name column (for per-class evaluation)
CLASS_COL = "variant_class"

# ──────────────────────── Load data ─────────────────────────
print(f"\nLoading {args.data} ...")
df = pd.read_csv(args.data)
print(f"  Rows: {len(df):,}  |  Columns: {df.shape[1]}")
print(f"  Benign: {(df[LABEL_COL]==0).sum():,}  |  Pathogenic: {(df[LABEL_COL]==1).sum():,}")

# Cardinalities
N_GENES    = df["gene_id"].nunique()          # 130
N_SYN      = df["syndrome_id"].nunique()      # 15
N_VCLASS   = df["variant_class_id"].nunique() # 11

print(f"  Genes: {N_GENES}  |  Syndromes: {N_SYN}  |  Variant classes: {N_VCLASS}")

# ───────────────────────── Split ────────────────────────────
print("\nSplitting 70 / 15 / 15 (stratified on label) ...")
idx = df.index.to_numpy()
labels = df[LABEL_COL].to_numpy()

idx_train, idx_tmp, y_train_s, y_tmp_s = train_test_split(
    idx, labels, test_size=0.30, stratify=labels, random_state=SEED)
idx_val, idx_test = train_test_split(
    idx_tmp, test_size=0.50, stratify=y_tmp_s, random_state=SEED)

train_df = df.loc[idx_train].reset_index(drop=True)
val_df   = df.loc[idx_val].reset_index(drop=True)
test_df  = df.loc[idx_test].reset_index(drop=True)

print(f"  Train: {len(train_df):,}  |  Val: {len(val_df):,}  |  Test: {len(test_df):,}")

# ──────────────────────── Scaler ────────────────────────────
scaler = StandardScaler()
train_df[SCALE_COLS] = scaler.fit_transform(train_df[SCALE_COLS])
val_df[SCALE_COLS]   = scaler.transform(val_df[SCALE_COLS])
test_df[SCALE_COLS]  = scaler.transform(test_df[SCALE_COLS])

# ──────────────────── Build tensors ─────────────────────────
CONT_COLS = SCALE_COLS + BINARY_COLS  # 7 + 11 = 18 continuous features

def to_tensors(frame):
    x_cont = torch.tensor(frame[CONT_COLS].values, dtype=torch.float32)
    x_gene  = torch.tensor(frame["gene_id"].values,         dtype=torch.long)
    x_syn   = torch.tensor(frame["syndrome_id"].values,     dtype=torch.long)
    x_vcls  = torch.tensor(frame["variant_class_id"].values, dtype=torch.long)
    y       = torch.tensor(frame[LABEL_COL].values,         dtype=torch.float32)
    return TensorDataset(x_cont, x_gene, x_syn, x_vcls, y)

train_ds = to_tensors(train_df)
val_ds   = to_tensors(val_df)
test_ds  = to_tensors(test_df)

train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=0)

# ─────────────────────── Architecture ───────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return torch.nn.functional.gelu(x + self.block(x))


class GenomeDLResNetV2(nn.Module):
    """
    Tabular ResNet for v2 hereditary-cancer variant classification.

    Inputs
    ------
    x_cont  : (B, 18)  — scaled continuous + binary features
    x_gene  : (B,)     — gene_id  ∈ [0, 129]
    x_syn   : (B,)     — syndrome_id ∈ [0, 14]
    x_vcls  : (B,)     — variant_class_id ∈ [0, 10]

    Output
    ------
    logit   : (B,)     — raw logit; apply sigmoid for probability
    """
    def __init__(
        self,
        n_cont:    int = 18,
        n_genes:   int = 130,
        n_syn:     int = 15,
        n_vclass:  int = 11,
        emb_gene:  int = 16,
        emb_syn:   int = 8,
        emb_vcls:  int = 8,
        hidden:    int = 256,
        n_blocks:  int = 2,
        dropout:   float = 0.2,
        head_drop: float = 0.1,
    ):
        super().__init__()
        self.gene_emb  = nn.Embedding(n_genes,  emb_gene)
        self.syn_emb   = nn.Embedding(n_syn,    emb_syn)
        self.vcls_emb  = nn.Embedding(n_vclass, emb_vcls)

        in_dim = n_cont + emb_gene + emb_syn + emb_vcls  # 18+16+8+8 = 50

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(head_drop),
            nn.Linear(64, 1),
        )

    def forward(self, x_cont, x_gene, x_syn, x_vcls):
        e = torch.cat([
            x_cont,
            self.gene_emb(x_gene),
            self.syn_emb(x_syn),
            self.vcls_emb(x_vcls),
        ], dim=1)
        h = self.input_proj(e)
        h = self.res_blocks(h)
        return self.head(h).squeeze(1)


# ───────────────────────── Model ────────────────────────────
model = GenomeDLResNetV2(
    n_cont   = len(CONT_COLS),  # 18
    n_genes  = N_GENES,
    n_syn    = N_SYN,
    n_vclass = N_VCLASS,
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nModel: GenomeDLResNetV2  |  Parameters: {n_params:,}")

# ─────────────────── Loss & optimiser ───────────────────────
n_neg = (train_df[LABEL_COL] == 0).sum()
n_pos = (train_df[LABEL_COL] == 1).sum()
pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
print(f"pos_weight = {pos_weight.item():.4f}  (n_neg={n_neg:,}, n_pos={n_pos:,})")

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

# ──────────────────── Evaluation helper ─────────────────────
def evaluate(loader, frame=None):
    """Returns (loss, auc_roc, auprc, all_probs, all_labels)."""
    model.eval()
    all_logits, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for x_cont, x_gene, x_syn, x_vcls, y in loader:
            x_cont, x_gene, x_syn, x_vcls, y = (
                x_cont.to(device), x_gene.to(device),
                x_syn.to(device),  x_vcls.to(device), y.to(device)
            )
            logits = model(x_cont, x_gene, x_syn, x_vcls)
            total_loss += criterion(logits, y).item() * len(y)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    all_logits = torch.cat(all_logits).numpy()
    all_labels = torch.cat(all_labels).numpy()
    probs = 1 / (1 + np.exp(-all_logits))

    loss  = total_loss / len(all_labels)
    auc   = roc_auc_score(all_labels, probs)
    auprc = average_precision_score(all_labels, probs)
    return loss, auc, auprc, probs, all_labels


# ─────────────────────── Training loop ──────────────────────
Path(args.outdir).mkdir(exist_ok=True)
best_val_auc = 0.0
best_ckpt    = {}

print(f"\n{'Epoch':>5}  {'Train Loss':>10}  {'Val Loss':>10}  {'Val AUC':>9}  {'Val AUPRC':>10}  {'LR':>8}  {'Time':>6}")
print("-" * 72)

for epoch in range(1, args.epochs + 1):
    t0 = time.time()
    model.train()
    train_loss = 0.0

    for x_cont, x_gene, x_syn, x_vcls, y in train_loader:
        x_cont, x_gene, x_syn, x_vcls, y = (
            x_cont.to(device), x_gene.to(device),
            x_syn.to(device),  x_vcls.to(device), y.to(device)
        )
        optimizer.zero_grad()
        logits = model(x_cont, x_gene, x_syn, x_vcls)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(y)

    scheduler.step()
    train_loss /= len(train_ds)

    val_loss, val_auc, val_auprc, _, _ = evaluate(val_loader)
    elapsed = time.time() - t0
    lr_now  = scheduler.get_last_lr()[0]

    print(f"{epoch:5d}  {train_loss:10.4f}  {val_loss:10.4f}  {val_auc:9.4f}  {val_auprc:10.4f}  {lr_now:8.2e}  {elapsed:5.1f}s")

    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_ckpt = {
            "epoch":    epoch,
            "val_auc":  val_auc,
            "val_auprc": val_auprc,
            "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
            # --- metadata needed by predict.py ---
            "cont_cols":    CONT_COLS,
            "scale_cols":   SCALE_COLS,
            "binary_cols":  BINARY_COLS,
            "scaler_mean":  scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "n_genes":      N_GENES,
            "n_syn":        N_SYN,
            "n_vclass":     N_VCLASS,
            "gene_to_id":   {g: int(i) for g, i in
                             zip(df["gene"].unique(), df["gene_id"].unique())},
            "syndrome_to_id": {s: int(i) for s, i in
                               zip(df["syndrome_group"].unique(), df["syndrome_id"].unique())},
            "class_to_id":  {c: int(i) for c, i in
                             zip(df["variant_class"].unique(), df["variant_class_id"].unique())},
        }

# ─────────────────── Reload best checkpoint ─────────────────
model.load_state_dict(best_ckpt["model_state_dict"])
print(f"\nBest val AUC {best_val_auc:.4f} at epoch {best_ckpt['epoch']}")

# ─────────────────────── Test evaluation ────────────────────
test_loss, test_auc, test_auprc, test_probs, test_labels = evaluate(test_loader)

print(f"\n{'='*60}")
print(f"TEST RESULTS")
print(f"{'='*60}")
print(f"  Loss  : {test_loss:.4f}")
print(f"  AUC   : {test_auc:.4f}")
print(f"  AUPRC : {test_auprc:.4f}")

# Binary predictions at 0.5 threshold
test_preds = (test_probs >= 0.5).astype(int)
print(f"\nClassification report (threshold=0.5):")
print(classification_report(test_labels, test_preds, target_names=["Benign", "Pathogenic"], digits=4))

# ─────────── Per-variant-class AUC (the honest metrics) ─────
print(f"\nPer-class AUC on test set:")
print(f"  {'Class':<25}  {'N':>7}  {'% Path':>7}  {'AUC':>7}  {'AUPRC':>7}")
print(f"  {'-'*60}")

test_df["_prob"]  = test_probs
test_df["_label"] = test_labels.astype(int)

HIGHLIGHT = {"missense", "splice_region"}

for cls in sorted(test_df[CLASS_COL].unique()):
    sub = test_df[test_df[CLASS_COL] == cls]
    n_cls  = len(sub)
    pct_p  = 100 * sub["_label"].mean()
    if sub["_label"].nunique() < 2:
        auc_cls  = float("nan")
        aprc_cls = float("nan")
    else:
        auc_cls  = roc_auc_score(sub["_label"], sub["_prob"])
        aprc_cls = average_precision_score(sub["_label"], sub["_prob"])
    flag = " <-- HEADLINE" if cls in HIGHLIGHT else ""
    print(f"  {cls:<25}  {n_cls:>7,}  {pct_p:>6.1f}%  {auc_cls:>7.4f}  {aprc_cls:>7.4f}{flag}")

# ─────────────────────── Save checkpoint ────────────────────
ckpt_path = Path(args.outdir) / "genome_dl_resnet_v2.pth"
# Add final test metrics to checkpoint
best_ckpt["test_auc"]   = test_auc
best_ckpt["test_auprc"] = test_auprc
best_ckpt["test_loss"]  = test_loss
torch.save(best_ckpt, ckpt_path)
print(f"\nCheckpoint saved → {ckpt_path}")

# ─────────────────── Save training metadata ─────────────────
meta = {
    "script": "train_resnet_v2.py",
    "data":   args.data,
    "n_train": len(train_df),
    "n_val":   len(val_df),
    "n_test":  len(test_df),
    "epochs":  args.epochs,
    "best_epoch": best_ckpt["epoch"],
    "val_auc":    best_ckpt["val_auc"],
    "val_auprc":  best_ckpt["val_auprc"],
    "test_auc":   test_auc,
    "test_auprc": test_auprc,
    "pos_weight": pos_weight.item(),
    "cont_cols":  CONT_COLS,
    "n_genes":    N_GENES,
    "n_syn":      N_SYN,
    "n_vclass":   N_VCLASS,
    "device":     str(device),
}
meta_path = Path(args.outdir) / "genome_dl_resnet_v2_meta.json"
with open(meta_path, "w") as f:
    json.dump(meta, f, indent=2)
print(f"Metadata saved  → {meta_path}")
print(f"\nDone.")
