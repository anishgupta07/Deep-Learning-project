"""
tests/test_stage0.py — GenomeDL v3 Stage 0 test suite
=======================================================
Run from repo root:
    python -m pytest tests/test_stage0.py -v

Tests:
  T1  No foreign-gene rows in clinvar_labelled_v3.csv
  T2  No stop-loss variants classed as 'missense'
  T3  panel.json exists and covers all 130 genes
  T4  Strand counts: 65 plus / 65 minus (or close, within 10 for data-derived)
  T5  Protein lengths are >= 50 (no junk) and <= 10000 for all genes
  T6  start_lost rows have grantham_score == 215.0 in v3_featured_snv.csv
  T7  Coding-strand one-hot columns present in v3_featured_snv.csv
  T8  Unknown gene raises ValueError
  T9  v3 row count is 148,825 - dropped_foreign (expect ~148,746)
  T10 Feature parity: 50 random rows recomputed == stored values
"""
import json, math, os, re, sys
import numpy as np
import pandas as pd
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data")

# ─── helpers ────────────────────────────────────────────────────────────────
def _load_v3():
    p = os.path.join(DATA, "clinvar_labelled_v3.csv")
    if not os.path.exists(p):
        pytest.skip("clinvar_labelled_v3.csv not found — run extract_v3.py first")
    return pd.read_csv(p, dtype={"chrom": str})

def _load_featured():
    p = os.path.join(DATA, "v3_featured_snv.csv")
    if not os.path.exists(p):
        pytest.skip("v3_featured_snv.csv not found — run build_v3_features.py first")
    return pd.read_csv(p)

def _load_panel():
    p = os.path.join(DATA, "panel.json")
    if not os.path.exists(p):
        pytest.skip("panel.json not found — run make_panel_json.py first")
    return json.load(open(p))

HGVS_GENE_RE = re.compile(r"\(([A-Z][A-Z0-9]+)\):c\.")

# ─── T1: No foreign-gene rows ────────────────────────────────────────────────
def test_no_foreign_gene_rows():
    df = _load_v3()
    foreign = 0
    for _, row in df.iterrows():
        m = HGVS_GENE_RE.search(str(row.get("hgvs_name", "")))
        if m and m.group(1) != row["gene"]:
            foreign += 1
    assert foreign == 0, f"{foreign} foreign-gene rows still present in v3 CSV"

# ─── T2: No stop-loss classed as missense ────────────────────────────────────
def test_no_stop_loss_as_missense():
    df = _load_v3()
    # stop-loss: ref_aa == '*' but classed as missense
    bad = df[(df["variant_class"] == "missense") & (df["ref_aa"] == "*")]
    assert len(bad) == 0, (
        f"{len(bad)} stop-loss variants still classed as 'missense'.\n"
        f"Sample:\n{bad[['gene','hgvs_name','ref_aa','alt_aa','variant_class']].head()}"
    )

# ─── T3: panel.json has 130 genes ────────────────────────────────────────────
def test_panel_json_completeness():
    panel = _load_panel()
    assert len(panel) == 130, f"panel.json has {len(panel)} genes, expected 130"
    for gene, entry in panel.items():
        for key in ("syndrome_group", "strand", "protein_length", "mane_transcript"):
            assert key in entry, f"panel.json missing '{key}' for gene {gene}"

# ─── T4: Strand counts ───────────────────────────────────────────────────────
def test_strand_distribution():
    panel = _load_panel()
    plus  = sum(1 for e in panel.values() if e["strand"] == "+")
    minus = sum(1 for e in panel.values() if e["strand"] == "-")
    # We expect ~65/65 but data-derived may differ slightly for sparse genes
    assert plus + minus >= 120, (
        f"Too many unknown strands: plus={plus}, minus={minus}, "
        f"unknown={130-plus-minus}"
    )
    # Soft check — allow ±10 around 65
    assert 55 <= plus <= 75,  f"Plus-strand count {plus} outside expected range 55-75"
    assert 55 <= minus <= 75, f"Minus-strand count {minus} outside expected range 55-75"

# ─── T5: Protein lengths sensible ────────────────────────────────────────────
def test_protein_lengths():
    panel = _load_panel()
    for gene, entry in panel.items():
        L = entry["protein_length"]
        assert L >= 50,    f"{gene}: protein_length={L} is suspiciously short (<50)"
        assert L <= 10000, f"{gene}: protein_length={L} is suspiciously long (>10000)"
    # Spot-check known lengths
    checks = {"BRCA1": 1863, "BRCA2": 3418, "TP53": 393, "VHL": 213}
    for gene, expected in checks.items():
        actual = panel.get(gene, {}).get("protein_length", 0)
        assert abs(actual - expected) <= 2, (
            f"{gene}: protein_length={actual}, expected ~{expected}"
        )

# ─── T6: start_lost severity in featured CSV ─────────────────────────────────
def test_start_lost_grantham():
    df = _load_featured()
    if "variant_class" not in df.columns:
        pytest.skip("variant_class column not in featured CSV")
    sl = df[df["variant_class"] == "start_lost"]
    if len(sl) == 0:
        pytest.skip("No start_lost rows in featured CSV")
    unique_g = sl["grantham_score"].unique()
    assert all(g == 215.0 for g in unique_g), (
        f"start_lost grantham_score has non-215.0 values: {unique_g}"
    )

# ─── T7: Coding-strand columns present ───────────────────────────────────────
def test_coding_strand_columns():
    df = _load_featured()
    expected = [f"cs_{side}_{b}" for side in ("ref", "alt") for b in "ACGT"]
    missing = [c for c in expected if c not in df.columns]
    assert not missing, f"Missing coding-strand columns: {missing}"

# ─── T8: Unknown gene raises ValueError ──────────────────────────────────────
def test_unknown_gene_error(tmp_path):
    sys.path.insert(0, os.path.join(ROOT, "src", "v3", "stage0"))
    try:
        from build_v3_features import featurize
    except ImportError:
        pytest.skip("build_v3_features not importable")

    # Create a mini CSV with an unknown gene
    mini_csv = tmp_path / "mini.csv"
    panel_path = os.path.join(DATA, "panel.json")
    if not os.path.exists(panel_path):
        pytest.skip("panel.json not found")

    # Pick a known gene for the valid row
    panel = json.load(open(panel_path))
    valid_gene = list(panel.keys())[0]
    mini_df = pd.DataFrame([{
        "variation_id": 1, "gene": "FAKEGENE999",
        "syndrome_group": "Other_hereditary", "chrom": "1",
        "pos": 100, "ref": "A", "alt": "C", "label": 0,
        "clinical_significance": "Benign", "review_status": "criteria provided",
        "stars": 1, "variant_class": "missense", "ref_aa": "A", "alt_aa": "V",
        "aa_pos": 10, "transcript": "", "hgvs_name": "", "last_evaluated": "",
        "on_primary_transcript": 1,
    }])
    mini_df.to_csv(mini_csv, index=False)

    with pytest.raises(ValueError, match="not in 130-gene panel"):
        featurize(str(mini_csv), panel_path, str(tmp_path / "out.csv"))

# ─── T9: Row count after stage0 fix ──────────────────────────────────────────
def test_v3_row_count():
    df = _load_v3()
    n = len(df)
    # Original was 148,825; expect 148,825 - 79 = 148,746 (± a few if extractor
    # was run on a different ClinVar snapshot)
    assert 147_000 <= n <= 149_000, (
        f"v3 labelled CSV has {n:,} rows, expected ~148,746"
    )
    print(f"\n  v3 labelled SNVs: {n:,}  (original v2 was 148,825)")

# ─── T10: Feature parity (50 random rows) ────────────────────────────────────
def test_feature_parity():
    df = _load_featured()
    if len(df) < 50:
        pytest.skip("Featured CSV has fewer than 50 rows")

    sys.path.insert(0, os.path.join(ROOT, "src", "v3", "stage0"))
    try:
        from build_v3_features import GRANTHAM, AA_PROPS, TRANSITIONS, COMP
    except ImportError:
        pytest.skip("build_v3_features not importable")

    panel_path = os.path.join(DATA, "panel.json")
    if not os.path.exists(panel_path):
        pytest.skip("panel.json not found")
    panel = json.load(open(panel_path))

    rng = np.random.default_rng(42)
    idx = rng.choice(len(df), size=50, replace=False)
    sample = df.iloc[idx]

    errors = []
    for _, row in sample.iterrows():
        gene  = row["gene"]
        vcls  = row["variant_class"]
        ra    = str(row.get("ref_aa", "."))
        aa_   = str(row.get("alt_aa", "."))
        pos   = int(row.get("aa_pos", 0))
        plen  = panel.get(gene, {}).get("protein_length", 1000)
        strand = panel.get(gene, {}).get("strand", "+")

        # Recompute grantham for missense
        if vcls == "missense" and ra in AA_PROPS and aa_ in AA_PROPS:
            expected_g = GRANTHAM.get((ra, aa_), 100.0)
            stored_g   = float(row.get("grantham_score", 0))
            if abs(expected_g - stored_g) > 0.5:
                errors.append(f"{gene} {vcls} {ra}->{aa_}: "
                              f"expected {expected_g}, stored {stored_g}")

        # Recompute rel_protein_pos for coding rows
        if vcls in ("missense", "nonsense", "start_lost", "stop_lost") and pos > 0:
            expected_rp = round(min(1.0, max(0.0, pos / plen)), 6)
            stored_rp   = float(row.get("rel_protein_pos", 0))
            if abs(expected_rp - stored_rp) > 0.01:
                errors.append(f"{gene} {vcls} pos={pos} plen={plen}: "
                              f"rel_pos expected {expected_rp:.4f}, stored {stored_rp:.4f}")

        # Recompute coding-strand one-hots for missense rows
        if "cs_ref_A" in row.index and vcls == "missense":
            r_base = str(row.get("ref","")).upper()
            if strand == "-" and r_base in "ACGT":
                r_base = r_base.translate(COMP)
            for b in "ACGT":
                expected_cs = int(r_base == b)
                stored_cs   = int(row.get(f"cs_ref_{b}", 0))
                if expected_cs != stored_cs:
                    errors.append(f"{gene} cs_ref_{b}: expected {expected_cs}, stored {stored_cs}")

    assert not errors, f"Feature parity failures ({len(errors)}):\n" + "\n".join(errors[:10])
