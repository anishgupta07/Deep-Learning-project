#!/usr/bin/env python3
"""
extract_v3.py — GenomeDL v3 ClinVar extractor (Stage 0 fixes applied)

Fixed defects vs v2 extractor:
  0.1 Drops 79 rows where the HGVS gene (extracted from Name "(GENE):c.")
      differs from the panel gene (e.g. MSH3 rows that describe DHFR).
  0.2 Adds "stop_lost" as variant class 12 (p.Ter214Leu style previously
      misclassified as missense). Now 12 variant classes total.

Usage (run from repo root, needs the raw ClinVar gz):
    python src/v3/stage0/extract_v3.py unused_files/variant_summary.txt.gz \\
           --outdir data --min-stars 1
"""
import argparse, csv, gzip, hashlib, json, os, re, sys, time
from collections import Counter, defaultdict

# ─────────────────────── gene panel ─────────────────────────────────────────
PANEL = {
 "Breast_Ovarian_HR_repair": ["BRCA1","BRCA2","PALB2","ATM","CHEK2","BARD1","BRIP1","RAD51C","RAD51D","RAD51B",
                              "NBN","MRE11","RAD50","XRCC2","RECQL","ABRAXAS1"],
 "Li_Fraumeni": ["TP53"],
 "Lynch_MMR_polymerase": ["MLH1","MSH2","MSH6","PMS2","EPCAM","MSH3","MLH3","PMS1","POLE","POLD1","MBD4"],
 "Polyposis_hamartoma": ["APC","MUTYH","NTHL1","AXIN2","GREM1","BMPR1A","SMAD4","STK11","PTEN","RNF43",
                         "FAN1","RPS20"],
 "Hereditary_gastric": ["CDH1","CTNNA1"],
 "Melanoma_skin_telomere": ["CDKN2A","CDK4","BAP1","MITF","POT1","ACD","TERF2IP","TERT"],
 "Endocrine_MEN": ["MEN1","RET","CDKN1B","CDC73","AIP","PRKAR1A"],
 "Paraganglioma_pheo": ["SDHA","SDHB","SDHC","SDHD","SDHAF2","TMEM127","MAX","VHL"],
 "Renal": ["FH","FLCN","MET","TSC1","TSC2"],
 "Neurocutaneous_rhabdoid_DICER1": ["NF1","NF2","SMARCB1","LZTR1","SUFU","PTCH1","PTCH2","SMARCA4","SMARCE1",
                                    "DICER1","SPRED1"],
 "Retinoblastoma_Wilms_overgrowth": ["RB1","WT1","DIS3L2","CDKN1C","REST","NSD1"],
 "Hematologic_predisposition": ["RUNX1","GATA2","CEBPA","DDX41","ETV6","ANKRD26","TINF2","DKC1","RTEL1","PARN",
                                "SAMD9","SAMD9L","SBDS"],
 "Fanconi_DNA_repair_chromosome_instability": ["FANCA","FANCB","FANCC","FANCD2","FANCE","FANCF","FANCG","FANCI",
                                               "FANCL","FANCM","SLX4","BLM","WRN","RECQL4"],
 "Xeroderma_NER": ["XPA","XPC","ERCC2","ERCC3","ERCC4","ERCC5","POLH","DDB2"],
 "Other_hereditary": ["HOXB13","EXT1","EXT2","PHOX2B","BUB1B","PTPN11","CEP57","TRIP13","CBL"],
}
GENE2SYN = {g: s for s, gs in PANEL.items() for g in gs}
assert len(GENE2SYN) == sum(len(v) for v in PANEL.values()), "duplicate gene in PANEL"

# 12 variant classes (added stop_lost in v3)
VARIANT_CLASSES = [
    "synonymous", "missense", "nonsense", "start_lost", "stop_lost",
    "splice_canonical", "splice_region", "intronic",
    "utr5", "utr3", "other_noncoding", "coding_no_protein_annot",
]
CLASS_TO_ID = {c: i for i, c in enumerate(VARIANT_CLASSES)}

# ─────────────────────── parsing helpers ────────────────────────────────────
AA3 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G","His":"H","Ile":"I",
       "Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V",
       "Ter":"*","Sec":"U"}
PROT_RE  = re.compile(r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=|\*|\?)\)")
FS_RE    = re.compile(r"\(p\.[A-Z][a-z]{2}\d+[A-Za-z]{0,3}fs")
TX_RE    = re.compile(r"^[NX][MR]_\d+\.\d+")
CDNA_RE  = re.compile(r":c\.([-*]?)(\d+)([+-]\d+)?")
# [FIX 0.1] HGVS gene extractor
HGVS_GENE_RE = re.compile(r"\(([A-Z][A-Z0-9]+)\):c\.")

STARS = {"practice guideline": 4, "reviewed by expert panel": 3,
         "criteria provided, multiple submitters, no conflicts": 2,
         "criteria provided, single submitter": 1,
         "criteria provided, conflicting classifications": 1,
         "criteria provided, conflicting interpretations": 1}

def stars(rs):
    return STARS.get(rs.strip().lower(), 0)

def classify(name):
    """Return (variant_class, ref_aa, alt_aa, aa_pos)."""
    m = PROT_RE.search(name)
    if m:
        ref3, pos, alt3 = m.groups()
        ref = AA3.get(ref3, "?")
        pos = int(pos)
        # [FIX 0.2] stop_lost: p.TerNNNAa (reference is stop codon)
        if ref3 == "Ter" and alt3 not in ("Ter", "*", "="):
            return "stop_lost", "*", AA3.get(alt3, "?"), pos
        if alt3 == "=":
            return "synonymous", ref, ref, pos
        if alt3 in ("Ter", "*"):
            return "nonsense", ref, "*", pos
        if alt3 == "?":
            return ("start_lost" if (ref3 == "Met" and pos == 1) else "protein_unknown"), ref, "?", pos
        if ref3 == "Met" and pos == 1:
            return "start_lost", ref, AA3.get(alt3, "?"), pos
        return "missense", ref, AA3.get(alt3, "?"), pos
    if FS_RE.search(name):
        return "frameshift", ".", ".", 0
    c = CDNA_RE.search(name)
    if c:
        sign, _, off = c.groups()
        if sign == "-":    return "utr5", ".", ".", 0
        if sign == "*":    return "utr3", ".", ".", 0
        if off:
            d = abs(int(off))
            if d <= 2:     return "splice_canonical", ".", ".", 0
            if d <= 8:     return "splice_region", ".", ".", 0
            return "intronic", ".", ".", 0
        return "coding_no_protein_annot", ".", ".", 0
    return "other_noncoding", ".", ".", 0

def label_from_sig(sig):
    s = sig.lower()
    path, ben, conf, unc = "pathogenic" in s, "benign" in s, "conflict" in s, "uncertain" in s
    if path and not (conf or unc or ben): return 1
    if ben and not (conf or unc or path): return 0
    if unc and not conf and not path and not ben: return 2   # VUS
    return -1                                                 # conflicting

def panel_gene(symbols):
    hits = [g for g in symbols.split(";") if g in GENE2SYN]
    return hits[0] if len(hits) == 1 else (None if not hits else "AMBIGUOUS")

def hgvs_gene(name):
    """Extract gene symbol from '(GENE):c.' in HGVS name, or None."""
    m = HGVS_GENE_RE.search(name)
    return m.group(1) if m else None

# ─────────────────────── main ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gz")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--min-stars", type=int, default=1)
    ap.add_argument("--indels", action="store_true")
    ap.add_argument("--hash", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    need = ["GeneSymbol","Assembly","OriginSimple","ClinicalSignificance","ReviewStatus","Type","Name",
            "Chromosome","PositionVCF","ReferenceAlleleVCF","AlternateAlleleVCF"]
    out_cols = ["variation_id","gene","syndrome_group","chrom","pos","ref","alt","label",
                "clinical_significance","review_status","stars","variant_class","ref_aa","alt_aa","aa_pos",
                "transcript","hgvs_name","last_evaluated"]
    files = {"labelled": "clinvar_labelled_v3.csv", "vus": "clinvar_vus_v3.csv",
             "conflict": "clinvar_conflicting_v3.csv"}
    if a.indels: files["indel"] = "clinvar_labelled_indels_v3.csv"
    fh = {k: open(os.path.join(a.outdir, v), "w", newline="", encoding="utf-8") for k, v in files.items()}
    wr = {k: csv.writer(f) for k, f in fh.items()}
    for w in wr.values(): w.writerow(out_cols)

    c = Counter()
    per_gene = defaultdict(Counter)
    seen = {}
    dropped_dupe_conflict = 0

    with gzip.open(a.gz, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline().lstrip("#").rstrip("\n").split("\t")
        col    = {n: i for i, n in enumerate(header)}
        miss   = [n for n in need if n not in col]
        if miss: sys.exit(f"Missing columns: {miss}")
        g = lambda p, n: p[col[n]] if n in col and col[n] < len(p) else ""

        for line in f:
            p = line.rstrip("\n").split("\t")
            c["rows"] += 1
            if len(p) <= col["AlternateAlleleVCF"]: continue
            gene = panel_gene(g(p, "GeneSymbol"))
            if gene is None: continue
            if gene == "AMBIGUOUS": c["skipped_overlapping_panel_genes"] += 1; continue
            c["panel_rows"] += 1
            if g(p, "Assembly") != "GRCh38": continue
            c["grch38"] += 1
            if "germline" not in g(p, "OriginSimple").lower(): continue
            c["germline"] += 1
            ref, alt = g(p, "ReferenceAlleleVCF").upper(), g(p, "AlternateAlleleVCF").upper()
            if ref in ("", "NA", "-") or alt in ("", "NA", "-"): c["no_vcf_alleles"] += 1; continue
            if ref == alt: c["ref_equals_alt_skipped"] += 1; continue
            snv = len(ref) == 1 and len(alt) == 1
            if not snv and not (a.indels and max(len(ref), len(alt)) <= 50):
                c["not_snv_skipped"] += 1; continue
            lab = label_from_sig(g(p, "ClinicalSignificance"))
            st  = stars(g(p, "ReviewStatus"))
            if lab in (0, 1) and st < a.min_stars: c["dropped_low_stars"] += 1; continue
            if lab == -1 and not snv: continue

            # [FIX 0.1] Drop rows where HGVS gene != panel gene
            name = g(p, "Name")
            hg = hgvs_gene(name)
            if hg is not None and hg != gene:
                c["dropped_foreign_gene_annotation"] += 1
                continue

            key = (g(p, "Chromosome"), g(p, "PositionVCF"), ref, alt, gene)
            if key in seen:
                c["duplicate_rows"] += 1
                if seen[key] != lab: dropped_dupe_conflict += 1
                continue
            seen[key] = lab

            vc, raa, aaa, apos = classify(name)
            row = [g(p, "VariationID"), gene, GENE2SYN[gene], key[0], key[1], ref, alt, lab,
                   g(p, "ClinicalSignificance"), g(p, "ReviewStatus"), st, vc, raa, aaa, apos,
                   (TX_RE.search(name) or [""])[0] if TX_RE.search(name) else "", name,
                   g(p, "LastEvaluated")]
            if not snv:
                if lab in (0, 1): wr["indel"].writerow(row); c["indel_labelled"] += 1
                continue
            dest = {0: "labelled", 1: "labelled", 2: "vus", -1: "conflict"}[lab]
            wr[dest].writerow(row)
            tag = {0: "benign", 1: "pathogenic", 2: "vus", -1: "conflict"}[lab]
            c[f"snv_{tag}"] += 1
            per_gene[gene][tag] += 1
            if lab in (0, 1): per_gene[gene][f"cls_{vc}"] += 1

    for f_ in fh.values(): f_.close()

    # ── post-pass: flag on_primary_transcript ─────────────────────────────────
    tx = defaultdict(Counter)
    for k in ("labelled", "vus"):
        path_ = os.path.join(a.outdir, files[k])
        if not os.path.exists(path_): continue
        with open(path_, newline="", encoding="utf-8") as fi:
            for r in csv.DictReader(fi):
                if r["transcript"]: tx[r["gene"]][r["transcript"]] += 1
    primary = {gn: cnt.most_common(1)[0][0] for gn, cnt in tx.items()}
    off_primary = 0
    for k, fn in files.items():
        path_ = os.path.join(a.outdir, fn)
        if not os.path.exists(path_): continue
        tmp = path_ + ".tmp"
        with open(path_, newline="", encoding="utf-8") as fi, \
             open(tmp, "w", newline="", encoding="utf-8") as fo:
            rd = csv.reader(fi); wo = csv.writer(fo)
            hdr = next(rd); wo.writerow(hdr + ["on_primary_transcript"])
            ig, it = hdr.index("gene"), hdr.index("transcript")
            for r in rd:
                flag = int(bool(r[it]) and r[it] == primary.get(r[ig]))
                if k == "labelled" and not flag: off_primary += 1
                wo.writerow(r + [flag])
        os.replace(tmp, path_)
    c["labelled_snv_off_primary_transcript"] = off_primary

    # ── manifest ─────────────────────────────────────────────────────────────
    st_ = os.stat(a.gz)
    gene_table = {gn: {"group": GENE2SYN[gn],
                        **{k: per_gene[gn][k] for k in ("benign","pathogenic","vus","conflict")}}
                  for gn in GENE2SYN}
    zero   = [gn for gn, v in gene_table.items() if v["benign"] + v["pathogenic"] == 0]
    sparse = [gn for gn, v in gene_table.items() if 0 < v["benign"] + v["pathogenic"] < 30]
    man = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "version": "v3",
        "input": {"file": os.path.basename(a.gz), "bytes": st_.st_size,
                  "mtime": time.strftime("%Y-%m-%d", time.localtime(st_.st_mtime))},
        "params": {"min_stars": a.min_stars, "indels": a.indels},
        "n_panel_genes": len(GENE2SYN),
        "counts": dict(c),
        "primary_transcripts": primary,
        "dropped_duplicate_label_conflicts": dropped_dupe_conflict,
        "genes_with_zero_labelled": zero,
        "genes_with_lt30_labelled": sparse,
        "per_gene": gene_table,
        "variant_classes": VARIANT_CLASSES,
        "class_to_id": CLASS_TO_ID,
        "defects_fixed": {
            "0.1_foreign_gene_rows_dropped": c["dropped_foreign_gene_annotation"],
            "0.2_stop_lost_class_added": True,
        }
    }
    json.dump(man, open(os.path.join(a.outdir, "manifest_v3.json"), "w"), indent=2)

    # ── console report ────────────────────────────────────────────────────────
    print("=" * 60)
    print("GenomeDL v3 Extractor — Stage 0 Results")
    print("=" * 60)
    print(f"Panel genes : {len(GENE2SYN)} across {len(PANEL)} groups")
    print(f"Variant classes: {len(VARIANT_CLASSES)} (added stop_lost)")
    print()
    for k in ["rows","panel_rows","grch38","germline",
              "dropped_foreign_gene_annotation",
              "snv_benign","snv_pathogenic","snv_vus","snv_conflict",
              "dropped_low_stars","duplicate_rows","ref_equals_alt_skipped",
              "labelled_snv_off_primary_transcript"]:
        print(f"  {k:<42} {c[k]:>10,}")
    total_labelled = c["snv_benign"] + c["snv_pathogenic"]
    print(f"\n  {'TOTAL LABELLED SNVs':<42} {total_labelled:>10,}")
    print(f"  {'FIX 0.1: foreign-gene rows dropped':<42} {c['dropped_foreign_gene_annotation']:>10,}")
    print(f"\nGenes with 0 labelled: {zero or 'none'}")
    print(f"Genes with <30 labelled: {sparse or 'none'}")
    print(f"\nWrote to: {a.outdir}/")
    print("=" * 60)

    # Class distribution for labelled SNVs
    print("\nVariant class distribution (labelled SNVs):")
    labelled_path = os.path.join(a.outdir, files["labelled"])
    if os.path.exists(labelled_path):
        cls_counts = Counter()
        path_counts = Counter()
        with open(labelled_path, newline="", encoding="utf-8") as fi:
            for r in csv.DictReader(fi):
                cls_counts[r["variant_class"]] += 1
                if r["label"] == "1":
                    path_counts[r["variant_class"]] += 1
        print(f"  {'Class':<30} {'Total':>8} {'Path':>8} {'%Path':>7}")
        print(f"  {'-'*58}")
        for cls in VARIANT_CLASSES:
            tot  = cls_counts.get(cls, 0)
            path = path_counts.get(cls, 0)
            pct  = 100 * path / tot if tot > 0 else 0
            if tot > 0:
                print(f"  {cls:<30} {tot:>8,} {path:>8,} {pct:>6.1f}%")

if __name__ == "__main__":
    main()
