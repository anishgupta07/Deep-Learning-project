#!/usr/bin/env python3
"""apply_fixes.py — Apply Stage 0 fixes 0.1 & 0.2 to existing CSV
===============================================================
Since re-running the full extractor requires the ~450MB gz (in archive/data/),
this script applies the same fixes directly to the archived or existing
clinvar_labelled_snv.csv:

  Fix 0.1: Drop 79 rows where HGVS gene != panel gene
  Fix 0.2: Reclassify stop-loss variants (p.Ter->Aa) from 'missense' to 'stop_lost'

Output: data/processed/clinvar_labelled_v3.csv
"""
import os, re
import pandas as pd

from genomedl.data.extract_variants import VARIANT_CLASSES
from genomedl.paths import CLINVAR_LABELLED_CSV, DATA_PROCESSED_DIR, PROJECT_ROOT

HGVS_GENE_RE = re.compile(r"\(([A-Z][A-Z0-9]+)\):c\.")

def main():
    archive_src = PROJECT_ROOT / "archive" / "data" / "clinvar_labelled_snv.csv"
    src = str(archive_src) if archive_src.exists() else str(CLINVAR_LABELLED_CSV)
    DATA_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dest = str(CLINVAR_LABELLED_CSV)
    print(f"Loading {src} ...")
    df = pd.read_csv(src, dtype={"chrom": str})
    n_orig = len(df)
    print(f"  Original rows: {n_orig:,}")

    # ── Fix 0.1: drop foreign-gene annotation rows ─────────────────────────
    def is_foreign(row):
        m = HGVS_GENE_RE.search(str(row.get("hgvs_name", "")))
        if m and m.group(1) != row["gene"]:
            return True
        return False

    foreign_mask = df.apply(is_foreign, axis=1)
    n_foreign = foreign_mask.sum()
    df = df[~foreign_mask].copy()
    print(f"\n[Fix 0.1] Dropped {n_foreign} foreign-gene annotation rows")
    print(f"  Remaining rows: {len(df):,}")

    # ── Fix 0.2: reclassify stop-loss from 'missense' to 'stop_lost' ──────
    # stop-loss: p.TerNNNAa — ref_aa is '*' and variant_class is 'missense'
    stop_loss_mask = (df["variant_class"] == "missense") & (df["ref_aa"] == "*")
    n_stoploss = stop_loss_mask.sum()
    df.loc[stop_loss_mask, "variant_class"] = "stop_lost"
    print(f"\n[Fix 0.2] Reclassified {n_stoploss} stop-loss variants from 'missense' to 'stop_lost'")

    # Verify no more stop-loss-as-missense
    still_bad = ((df["variant_class"] == "missense") & (df["ref_aa"] == "*")).sum()
    assert still_bad == 0, f"Still {still_bad} stop-loss rows classed as missense!"

    # ── Print class distribution ───────────────────────────────────────────
    print(f"\nFinal row count: {len(df):,}  (removed {n_orig - len(df):,} rows)")
    print(f"\nVariant class distribution:")
    for vc in VARIANT_CLASSES:
        sub  = df[df["variant_class"] == vc]
        path = (sub["label"] == 1).sum()
        if len(sub) > 0:
            print(f"  {vc:<30} {len(sub):>8,}  path={path:,} ({100*path/len(sub):.1f}%)")

    df.to_csv(dest, index=False)
    print(f"\nWritten: {dest}")
    print(f"  Rows: {len(df):,}  (expected ~148,746)")

if __name__ == "__main__":
    main()
