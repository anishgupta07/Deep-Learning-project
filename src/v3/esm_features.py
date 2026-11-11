"""Stage 1: ESM-2 Protein Language Model Feature Extraction and Masked-Marginal Scoring.

Computes masked-marginal log-likelihood ratio:
    esm_llr = log P(alt_aa | context) - log P(ref_aa | context)
with the variant position masked in the primary protein sequence.

Extracts per-position representation vector (masked position embedding) for fusion model.
Caches per-(gene, transcript, aa_pos) predictions in data/esm_cache.parquet.
Produces data/v3_featured_snv.csv with esm_llr, esm_available, and data/esm_embeddings.pt.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Safeguard against strict dependency version mismatch in transformers when huggingface-hub>=2.0 is installed
import types
if "transformers.dependency_versions_check" not in sys.modules:
    _dvc = types.ModuleType("transformers.dependency_versions_check")
    _dvc.dep_version_check = lambda *a, **k: None
    _dvc.require_version_core = lambda *a, **k: None
    _dvc.require_version = lambda *a, **k: None
    sys.modules["transformers.dependency_versions_check"] = _dvc

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

try:
    from src.v3.fetch_proteins import fetch_all_panel_sequences
except ImportError:
    from fetch_proteins import fetch_all_panel_sequences

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("esm_features")

STANDARD_AAS = ["A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
MAX_ESM_CONTEXT = 1022  # 1024 - <cls> - <eos>


def get_centered_window(seq: str, aa_pos: int, max_len: int = MAX_ESM_CONTEXT) -> Tuple[str, int]:
    """Get a sequence window centered on aa_pos (1-indexed) within max_len.
    
    Returns:
        (window_subseq, 0_indexed_pos_in_window)
    """
    p = aa_pos - 1
    total_len = len(seq)
    if total_len <= max_len:
        return seq, p

    start = max(0, p - max_len // 2)
    end = min(total_len, start + max_len)
    if end - start < max_len:
        start = max(0, end - max_len)
    return seq[start:end], p - start


def load_fasta(fasta_path: Path) -> Dict[str, str]:
    """Load protein sequences from FASTA file."""
    if not fasta_path.exists():
        logger.info(f"FASTA {fasta_path} not found. Fetching via UniProt REST API...")
        return fetch_all_panel_sequences()

    gene_seqs = {}
    curr_gene = ""
    curr_seq = []
    with open(fasta_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                if curr_gene:
                    gene_seqs[curr_gene] = "".join(curr_seq)
                curr_gene = line[1:].split()[0]
                curr_seq = []
            else:
                curr_seq.append(line)
        if curr_gene:
            gene_seqs[curr_gene] = "".join(curr_seq)
    return gene_seqs


def load_or_reconstruct_embeddings(
    featured_csv: Path,
    cache_parquet: Path,
    embeddings_pt: Optional[Path] = None,
) -> torch.Tensor:
    """Load precomputed embeddings or reconstruct from esm_cache.parquet.

    Avoids silent fallbacks to dummy zeros on fresh clones where data/*.pt
    is excluded by .gitignore (>100MB limit).
    """
    if embeddings_pt is not None and embeddings_pt.exists():
        logger.info(f"Loading cached embeddings tensor from {embeddings_pt}...")
        return torch.load(embeddings_pt, map_location="cpu", weights_only=True)

    if not featured_csv.exists() or not cache_parquet.exists():
        raise FileNotFoundError(
            f"Cannot load or reconstruct embeddings: {featured_csv} or {cache_parquet} is missing."
        )

    logger.info(f"Reconstructing embeddings matrix from {cache_parquet} for {featured_csv}...")
    featured_df = pd.read_csv(featured_csv, usecols=["gene", "aa_pos", "esm_available"])
    cache_df = pd.read_parquet(cache_parquet, columns=["gene", "aa_pos", "embedding"])

    cache_map = {
        (r.gene, int(r.aa_pos)): r.embedding for r in cache_df.itertuples(index=False)
    }

    matrix = np.zeros((len(featured_df), 480), dtype=np.float16)
    for i, (g, p, av) in enumerate(
        zip(featured_df["gene"], featured_df["aa_pos"], featured_df["esm_available"])
    ):
        if av == 1 and pd.notna(p) and p > 0:
            key = (g, int(p))
            if key in cache_map:
                matrix[i] = cache_map[key].astype(np.float16)

    emb_tensor = torch.from_numpy(matrix)
    logger.info(f"Reconstructed embeddings tensor shape {emb_tensor.shape} ({emb_tensor.dtype}).")

    if embeddings_pt is not None:
        try:
            embeddings_pt.parent.mkdir(parents=True, exist_ok=True)
            torch.save(emb_tensor, embeddings_pt)
            logger.info(f"Cached reconstructed tensor to {embeddings_pt}.")
        except Exception as e:
            logger.warning(f"Could not cache reconstructed tensor to {embeddings_pt}: {e}")

    return emb_tensor


def run_esm_inference_for_positions(
    unique_positions: List[Tuple[str, str, int, str]],  # (gene, transcript, aa_pos, ref_aa)
    gene_seqs: Dict[str, str],
    model_name: str,
    device: str,
    batch_size: int = 16,
) -> Dict[Tuple[str, int], Tuple[Dict[str, float], np.ndarray]]:
    """Run ESM masked language model for unique (gene, aa_pos) pairs.
    
    Returns:
        dict: (gene, aa_pos) -> (log_probs_dict, embedding_vector)
    """
    dtype = torch.float16 if device == "cuda" else torch.float32
    logger.info(f"Loading {model_name} on {device} ({dtype})...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMaskedLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()

    # Get token IDs for standard amino acids and stop
    aa_tokens = {aa: tokenizer.convert_tokens_to_ids(aa) for aa in STANDARD_AAS}
    stop_id = tokenizer.convert_tokens_to_ids("*")

    # Filter positions where gene exists and residue is in bounds
    valid_items = []
    for gene, transcript, aa_pos, ref_aa in unique_positions:
        if gene not in gene_seqs:
            continue
        seq = gene_seqs[gene]
        if aa_pos < 1 or aa_pos > len(seq):
            continue
        valid_items.append((gene, aa_pos, transcript, ref_aa))

    logger.info(f"Computing ESM embeddings for {len(valid_items)} unique positions in batches of {batch_size}...")
    results = {}

    for i in range(0, len(valid_items), batch_size):
        batch = valid_items[i : i + batch_size]
        batch_seqs = []
        batch_mask_indices = []

        for gene, aa_pos, transcript, ref_aa in batch:
            seq = gene_seqs[gene]
            window, pos_in_win = get_centered_window(seq, aa_pos)
            batch_seqs.append(window)
            batch_mask_indices.append(pos_in_win + 1)  # +1 for <cls>

        # Tokenize batch
        inputs = tokenizer(batch_seqs, padding=True, return_tensors="pt").to(device)
        for b_idx, m_idx in enumerate(batch_mask_indices):
            inputs["input_ids"][b_idx, m_idx] = tokenizer.mask_token_id

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Logits: [batch_size, seq_len, vocab_size]
            logits = outputs.logits
            # Hidden: [batch_size, seq_len, hidden_dim]
            hidden = outputs.hidden_states[-1]

        for b_idx, (gene, aa_pos, transcript, ref_aa) in enumerate(batch):
            m_idx = batch_mask_indices[b_idx]
            pos_logits = logits[b_idx, m_idx]
            pos_log_probs = torch.log_softmax(pos_logits, dim=-1)
            pos_hidden = hidden[b_idx, m_idx].detach().cpu().to(torch.float32).numpy()

            log_p_dict = {aa: pos_log_probs[tok_id].item() for aa, tok_id in aa_tokens.items()}
            log_p_dict["*"] = pos_log_probs[stop_id].item()
            results[(gene, aa_pos)] = (log_p_dict, pos_hidden)

        if (i // batch_size + 1) % 50 == 0 or i + batch_size >= len(valid_items):
            logger.info(f"  Processed {min(i + batch_size, len(valid_items))}/{len(valid_items)} positions...")

    return results


def build_or_load_esm_cache(
    df: pd.DataFrame,
    gene_seqs: Dict[str, str],
    cache_path: Path,
    model_name: str,
    device: str,
    batch_size: int = 16,
    recompute: bool = False,
) -> pd.DataFrame:
    """Build or load the parquet cache of ESM predictions and embeddings."""
    if cache_path.exists() and not recompute:
        logger.info(f"Loading cached ESM predictions from {cache_path}...")
        return pd.read_parquet(cache_path)

    # Find unique protein positions
    prot_mask = df["variant_class"].isin(["missense", "stop_lost", "start_lost"])
    prot_df = df[prot_mask][["gene", "transcript", "aa_pos", "ref_aa"]].drop_duplicates()
    unique_positions = [
        (str(r.gene), str(r.transcript) if pd.notna(r.transcript) else "", int(r.aa_pos), str(r.ref_aa))
        for _, r in prot_df.iterrows()
        if pd.notna(r.aa_pos)
    ]

    logger.info(f"Identified {len(unique_positions)} unique protein positions across {len(prot_df)} variants.")
    esm_data = run_esm_inference_for_positions(unique_positions, gene_seqs, model_name, device, batch_size)

    # Convert to DataFrame
    cache_records = []
    for (gene, aa_pos), (log_p_dict, emb) in esm_data.items():
        rec = {
            "gene": gene,
            "aa_pos": aa_pos,
            "embedding": emb.tolist(),
        }
        for aa, val in log_p_dict.items():
            rec[f"log_p_{aa}"] = val
        cache_records.append(rec)

    cache_df = pd.DataFrame(cache_records)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_df.to_parquet(cache_path, index=False)
    logger.info(f"Saved ESM cache with {len(cache_df)} entries to {cache_path}.")
    return cache_df


def generate_v3_esm_features(
    input_csv: Path = PROJECT_ROOT / "data" / "clinvar_labelled_v3.csv",
    featured_csv: Path = PROJECT_ROOT / "data" / "v3_featured_snv.csv",
    protein_fasta: Path = PROJECT_ROOT / "data" / "panel_proteins.fasta",
    cache_path: Path = PROJECT_ROOT / "data" / "esm_cache.parquet",
    embeddings_out: Path = PROJECT_ROOT / "data" / "esm_embeddings.pt",
    model_name: str = "facebook/esm2_t12_35M_UR50D",
    device: Optional[str] = None,
    batch_size: int = 16,
    recompute: bool = False,
) -> Tuple[pd.DataFrame, torch.Tensor]:
    """Generate ESM features and update featured dataset."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info(f"Starting Stage 1 ESM pipeline (device={device}, model={model_name})...")
    gene_seqs = load_fasta(protein_fasta)
    logger.info(f"Loaded {len(gene_seqs)} protein sequences.")

    # Load labelled data
    df = pd.read_csv(input_csv, low_memory=False)
    logger.info(f"Loaded {len(df)} variants from {input_csv}.")

    # Build or load cache
    cache_df = build_or_load_esm_cache(df, gene_seqs, cache_path, model_name, device, batch_size, recompute)

    # Index cache by (gene, aa_pos)
    cache_dict = {}
    for _, row in cache_df.iterrows():
        key = (str(row["gene"]), int(row["aa_pos"]))
        emb = np.array(row["embedding"], dtype=np.float32)
        log_probs = {col.replace("log_p_", ""): float(row[col]) for col in cache_df.columns if col.startswith("log_p_")}
        cache_dict[key] = (log_probs, emb)

    # Load existing featured dataset if available, or build base
    if featured_csv.exists():
        out_df = pd.read_csv(featured_csv, low_memory=False)
    else:
        out_df = df.copy()

    emb_dim = len(next(iter(cache_dict.values()))[1]) if cache_dict else 480
    esm_llr_col = np.zeros(len(out_df), dtype=np.float32)
    esm_avail_col = np.zeros(len(out_df), dtype=np.int32)
    embeddings_matrix = np.zeros((len(out_df), emb_dim), dtype=np.float16)

    matched_residues = 0
    mismatched_residues = 0
    protein_count = 0

    for idx, row in df.iterrows():
        vclass = str(row["variant_class"])
        if vclass not in ["missense", "stop_lost", "start_lost"]:
            continue

        protein_count += 1
        gene = str(row["gene"])
        ref_aa = str(row["ref_aa"]) if pd.notna(row["ref_aa"]) else ""
        alt_aa = str(row["alt_aa"]) if pd.notna(row["alt_aa"]) else ""
        aa_pos = int(row["aa_pos"]) if pd.notna(row["aa_pos"]) else -1

        if gene not in gene_seqs or aa_pos < 1:
            continue

        seq = gene_seqs[gene]
        if vclass == "stop_lost":
            # stop_lost occurs at termination codon position len(seq) + 1 with ref_aa in ['*', 'Ter']
            if aa_pos == len(seq) + 1 and ref_aa in ["*", "Ter"]:
                matched_residues += 1
                # Stop lost is a definitive translation extension; ESM tokenizer maps '*' to <unk>,
                # so downstream pipeline treats it with rule-based LoF and high biochemical penalties.
                continue
            elif aa_pos <= len(seq) and ref_aa in ["*", "Ter"]:
                matched_residues += 1
                continue

        if aa_pos > len(seq):
            mismatched_residues += 1
            continue

        actual_seq_aa = seq[aa_pos - 1]
        # Verify sequence match
        if actual_seq_aa != ref_aa and ref_aa not in ["Ter", "*"]:
            mismatched_residues += 1
            continue

        matched_residues += 1
        key = (gene, aa_pos)
        if key in cache_dict:
            log_probs, emb = cache_dict[key]
            # alt_aa lookup (if stop, check '*')
            alt_key = "*" if alt_aa in ["Ter", "*"] else alt_aa
            ref_key = "*" if ref_aa in ["Ter", "*"] else ref_aa

            if alt_key in log_probs and ref_key in log_probs:
                llr = log_probs[alt_key] - log_probs[ref_key]
                esm_llr_col[idx] = llr
                esm_avail_col[idx] = 1
                embeddings_matrix[idx] = emb.astype(np.float16)

    match_rate = matched_residues / protein_count if protein_count > 0 else 0.0
    logger.info(f"--- ESM Sequence Validation ---")
    logger.info(f"Protein variants evaluated: {protein_count}")
    logger.info(f"Residue matches:            {matched_residues} ({match_rate * 100:.2f}%)")
    logger.info(f"Residue mismatches:         {mismatched_residues} ({mismatched_residues / protein_count * 100:.2f}%)")
    logger.info(f"Variants with ESM features:  {esm_avail_col.sum()}")

    out_df["esm_llr"] = esm_llr_col
    out_df["esm_available"] = esm_avail_col

    featured_csv.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(featured_csv, index=False)
    logger.info(f"Saved updated dataset with {len(out_df.columns)} columns to {featured_csv}.")

    emb_tensor = torch.from_numpy(embeddings_matrix)
    torch.save(emb_tensor, embeddings_out)
    logger.info(f"Saved embeddings tensor shape {emb_tensor.shape} ({emb_tensor.dtype}) to {embeddings_out}.")

    return out_df, emb_tensor


def parse_args():
    parser = argparse.ArgumentParser(description="Extract ESM features for GenomeDL v3")
    parser.add_argument("--input_csv", type=Path, default=PROJECT_ROOT / "data" / "clinvar_labelled_v3.csv")
    parser.add_argument("--featured_csv", type=Path, default=PROJECT_ROOT / "data" / "v3_featured_snv.csv")
    parser.add_argument("--protein_fasta", type=Path, default=PROJECT_ROOT / "data" / "panel_proteins.fasta")
    parser.add_argument("--cache_path", type=Path, default=PROJECT_ROOT / "data" / "esm_cache.parquet")
    parser.add_argument("--embeddings_out", type=Path, default=PROJECT_ROOT / "data" / "esm_embeddings.pt")
    parser.add_argument("--t30", action="store_true", help="Use 150M parameter esm2_t30 model instead of t12_35M")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--recompute", action="store_true", help="Recompute cache from scratch")
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    model = "facebook/esm2_t30_150M_UR50D" if args.t30 else "facebook/esm2_t12_35M_UR50D"
    generate_v3_esm_features(
        input_csv=args.input_csv,
        featured_csv=args.featured_csv,
        protein_fasta=args.protein_fasta,
        cache_path=args.cache_path,
        embeddings_out=args.embeddings_out,
        model_name=model,
        device=args.device,
        batch_size=args.batch_size,
        recompute=args.recompute,
    )
