"""Tests for Stage 1: ESM-2 Feature Extraction and Caching.

Covers:
- Sequence match rate >= 95%
- Cache structure and contents in data/esm_cache.parquet
- Zero-filling for non-protein rows
- Valid LLR and availability for known pathogenic missense variants
- Alignment and shape of data/esm_embeddings.pt
- Windowing logic for proteins > 1022 tokens
"""

import numpy as np
import pandas as pd
import pytest
import torch

from genomedl.features.esm_features import get_centered_window, load_fasta, load_or_reconstruct_embeddings
from genomedl.paths import (
    ESM_CACHE_PARQUET as CACHE_PARQUET,
    ESM_EMBEDDINGS_PT as EMBEDDINGS_PT,
    FEATURED_SNV_CSV as FEATURED_CSV,
    PANEL_FASTA as FASTA_PATH,
)


def test_fasta_loaded():
    """Verify FASTA contains all 130 panel genes."""
    assert FASTA_PATH.exists(), f"FASTA file not found at {FASTA_PATH}"
    gene_seqs = load_fasta(FASTA_PATH)
    assert len(gene_seqs) == 130, f"Expected 130 genes in FASTA, got {len(gene_seqs)}"
    assert "BRCA1" in gene_seqs
    assert len(gene_seqs["BRCA1"]) == 1863
    assert "TP53" in gene_seqs
    assert len(gene_seqs["TP53"]) == 393


def test_esm_cache_integrity():
    """Verify data/esm_cache.parquet structure and size."""
    assert CACHE_PARQUET.exists(), f"Cache not found at {CACHE_PARQUET}"
    cache_df = pd.read_parquet(CACHE_PARQUET)
    assert len(cache_df) >= 7000, f"Expected >= 7000 cached positions, got {len(cache_df)}"
    assert "gene" in cache_df.columns
    assert "aa_pos" in cache_df.columns
    assert "embedding" in cache_df.columns
    assert "log_p_A" in cache_df.columns
    assert "log_p_C" in cache_df.columns
    assert "log_p_Y" in cache_df.columns
    assert "log_p_*" in cache_df.columns

    # Check embedding dimension
    sample_emb = cache_df.iloc[0]["embedding"]
    assert len(sample_emb) == 480, f"Expected 480-dim embedding for esm2_t12, got {len(sample_emb)}"


def test_v3_featured_columns_and_rows():
    """Verify v3_featured_snv.csv schema and row count."""
    assert FEATURED_CSV.exists(), f"Featured CSV not found at {FEATURED_CSV}"
    df = pd.read_csv(FEATURED_CSV, low_memory=False)
    assert len(df) == 148747, f"Expected 148,747 rows, got {len(df)}"
    assert "esm_llr" in df.columns, "esm_llr column missing"
    assert "esm_available" in df.columns, "esm_available column missing"
    assert len(df.columns) == 49, f"Expected 49 columns, got {len(df.columns)}"


def test_esm_sequence_match_rate():
    """Verify ref_aa sequence match rate >= 95% for primary protein variants."""
    df = pd.read_csv(FEATURED_CSV, low_memory=False)
    prot_mask = df["variant_class"].isin(["missense", "stop_lost", "start_lost"])
    prot_df = df[prot_mask]
    total_prot = len(prot_df)
    avail_prot = (prot_df["esm_available"] == 1).sum()
    rate = avail_prot / total_prot
    print(f"\n[Test] Protein variants: {total_prot}, ESM available: {avail_prot}, Match rate: {rate * 100:.2f}%")
    assert rate >= 0.95, f"Match rate {rate * 100:.2f}% is below 95% threshold"


def test_non_protein_esm_zero():
    """Non-protein variants must have esm_available == 0 and esm_llr == 0.0."""
    df = pd.read_csv(FEATURED_CSV, low_memory=False)
    non_prot = df[~df["variant_class"].isin(["missense", "stop_lost", "start_lost"])]
    assert (non_prot["esm_available"] == 0).all(), "Found non-protein rows with esm_available != 0"
    assert np.allclose(non_prot["esm_llr"].values, 0.0), "Found non-protein rows with non-zero esm_llr"


def test_known_pathogenic_missense():
    """BRCA1 p.Cys61Gly (17:43106487 A>C) must have esm_available=1 and negative esm_llr."""
    df = pd.read_csv(FEATURED_CSV, low_memory=False)
    brca1_c61g = df[(df["gene"] == "BRCA1") & (df["aa_pos"] == 61) & (df["alt_aa"] == "G")]
    assert len(brca1_c61g) > 0, "BRCA1 C61G variant not found in dataset"
    row = brca1_c61g.iloc[0]
    assert row["esm_available"] == 1, "BRCA1 C61G should have esm_available=1"
    assert row["esm_llr"] < 0, f"Expected negative LLR for disruptive C61G substitution, got {row['esm_llr']}"


def test_embeddings_file():
    """Verify esm_embeddings tensor shape, dtype, and reconstruction from parquet."""
    emb = load_or_reconstruct_embeddings(FEATURED_CSV, CACHE_PARQUET, EMBEDDINGS_PT)
    assert emb.shape == (148747, 480), f"Expected shape (148747, 480), got {emb.shape}"
    assert emb.dtype == torch.float16, f"Expected float16 dtype, got {emb.dtype}"


def test_windowing_long_proteins():
    """Verify centered windowing logic for long proteins (>1022 residues)."""
    seq = "M" * 3418  # BRCA2 length
    # Test near N-terminus
    w1, p1 = get_centered_window(seq, 10)
    assert len(w1) == 1022
    assert p1 == 9

    # Test near C-terminus
    w2, p2 = get_centered_window(seq, 3410)
    assert len(w2) == 1022
    assert p2 == 3409 - (3418 - 1022)

    # Test center
    w3, p3 = get_centered_window(seq, 1709)
    assert len(w3) == 1022
    assert p3 == 1022 // 2


def test_portability_and_incomplete_panel_failure(monkeypatch):
    """Default CLI/module paths must be PROJECT_ROOT-anchored, and incomplete protein fetches must fail loudly."""
    import inspect
    import urllib.request
    from genomedl.data import fetch_proteins
    from genomedl.features import esm_features

    sig = inspect.signature(esm_features.generate_v3_esm_features)
    assert sig.parameters["input_csv"].default.is_absolute()
    assert fetch_proteins.PANEL_JSON.is_absolute()

    monkeypatch.setattr(fetch_proteins, "fetch_uniprot_batch", lambda batch: ">sp|P38398|BRCA1_HUMAN GN=BRCA1\nMDLSALR\n")
    monkeypatch.setattr(urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline test")))
    with pytest.raises(RuntimeError, match="Incomplete protein panel"):
        fetch_proteins.fetch_all_panel_sequences(force_download=True)
