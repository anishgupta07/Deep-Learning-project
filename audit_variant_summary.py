"""
Streaming audit of ClinVar variant_summary.txt.gz against the GenomeDL builder.
Uses only the standard library. Prints a short summary you can paste back.

Usage:
    python audit_variant_summary.py variant_summary.txt.gz build_pan_cancer_germline_dataset.py
"""
import gzip
import re
import sys
from collections import Counter

if len(sys.argv) != 3:
    sys.exit("Usage: python audit_variant_summary.py <variant_summary.txt.gz> <build_script.py>")

gz_path, build_path = sys.argv[1], sys.argv[2]

# Gene list copied from the builder's CANCER_GENES block
src = open(build_path, encoding="utf-8").read()
block = src.split("CANCER_GENES = {", 1)[1].split("\n}\n", 1)[0]
GENES = set(re.findall(r"^\s+'([A-Z0-9]+)':\s*\(", block, flags=re.M))
print(f"Genes parsed from builder: {len(GENES)}")

needed = ["GeneSymbol", "Assembly", "OriginSimple", "ClinicalSignificance", "ReviewStatus",
          "Type", "Name", "PositionVCF", "ReferenceAlleleVCF", "AlternateAlleleVCF"]

c = Counter()
review_by_label = {0: Counter(), 1: Counter()}
has_protein = {0: Counter(), 1: Counter()}
protein_re = re.compile(r"\(p\.[A-Z][a-z]{2}\d+([A-Z][a-z]{2}|=)\)")

with gzip.open(gz_path, "rt", encoding="utf-8", errors="ignore") as f:
    header = f.readline().lstrip("#").strip().split("\t")
    col = {n: i for i, n in enumerate(header)}
    missing = [n for n in needed if n not in col]
    print("Missing columns:", missing if missing else "none")
    if missing:
        print("Actual header:", header)
        sys.exit(1)
    for line in f:
        p = line.rstrip("\n").split("\t")
        c["total_rows"] += 1
        if len(p) <= col["PositionVCF"]:
            continue
        if p[col["GeneSymbol"]] not in GENES:
            continue
        c["panel_gene_rows"] += 1
        if p[col["Assembly"]] != "GRCh38":
            continue
        c["grch38"] += 1
        if "germline" not in p[col["OriginSimple"]].lower():
            continue
        c["germline"] += 1
        ref, alt = p[col["ReferenceAlleleVCF"]].upper(), p[col["AlternateAlleleVCF"]].upper()
        if len(ref) != 1 or len(alt) != 1:
            c["not_snv"] += 1
            continue
        c["snv"] += 1
        sig = p[col["ClinicalSignificance"]].lower()
        path, ben = "pathogenic" in sig, "benign" in sig
        conf, unc = "conflict" in sig, "uncertain" in sig
        if path and not (conf or unc or ben):
            label = 1
        elif ben and not (conf or unc or path):
            label = 0
        elif unc and not conf:
            label = 2
        else:
            c["conflicting_or_other_snv"] += 1
            continue
        c[f"label_{label}"] += 1
        if label in (0, 1):
            review_by_label[label][p[col["ReviewStatus"]]] += 1
            has_protein[label][bool(protein_re.search(p[col["Name"]]))] += 1

print("\n=== Filter funnel (README claims: 119,045 clean = 100,997 benign + 18,048 pathogenic; 135,839 VUS) ===")
for k in ["total_rows", "panel_gene_rows", "grch38", "germline", "not_snv", "snv",
          "conflicting_or_other_snv", "label_0", "label_1", "label_2"]:
    print(f"{k:28s} {c[k]:>10,}")

print("\n=== ReviewStatus of labelled benign(0) / pathogenic(1) SNVs ===")
for lab in (0, 1):
    tot = sum(review_by_label[lab].values())
    print(f"label {lab} (n={tot:,})")
    for k, v in review_by_label[lab].most_common():
        print(f"   {v:>8,} ({100*v/max(tot,1):4.1f}%)  {k}")

print("\n=== Labelled SNVs with a parsable protein change in Name ===")
for lab in (0, 1):
    t = sum(has_protein[lab].values())
    print(f"label {lab}: with protein change {has_protein[lab][True]:,} / {t:,}")