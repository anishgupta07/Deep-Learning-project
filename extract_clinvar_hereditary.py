#!/usr/bin/env python3
"""
Hereditary-cancer ClinVar dataset extractor (standard library only, streams the gz).

Input : ClinVar variant_summary.txt.gz  (GRCh38 rows used)
Output: <outdir>/
    clinvar_labelled_snv.csv      benign(0) / pathogenic(1) SNVs  -> training/eval set
    clinvar_vus_snv.csv           uncertain-significance SNVs     -> scoring only, never train
    clinvar_conflicting_snv.csv   conflicting SNVs                -> excluded from training
    clinvar_labelled_indels.csv   benign/pathogenic small indels  (only with --indels)
    manifest.json                 parameters, input file info, counts, per-gene table

Key design choices (fixing the audit findings):
  * hereditary predisposition genes only (no somatic-only drivers such as KRAS/EGFR/BRAF)
  * ReviewStatus kept, plus --min-stars filter (default 1 = require assertion criteria)
  * variant_class derived from the HGVS Name (missense / nonsense / synonymous /
    start_lost (incl. p.Met1Val) / splice_canonical / splice_region / intronic / utr5 / utr3 / ...),
    so you can report metrics PER CLASS and treat missense as the primary task
  * protein length is NOT hand-typed; aa_pos is raw. Join real lengths (MANE/UniProt) later.
  * transcript + on_primary_transcript columns: use on_primary_transcript==1 for any position-based analysis
  * every gene in the panel is reported, including genes with zero rows
Usage:
  python extract_clinvar_hereditary.py variant_summary.txt.gz --outdir out --min-stars 1 --indels
"""
import argparse, csv, gzip, hashlib, json, os, re, sys, time
from collections import Counter, defaultdict

# ----------------------------- gene panel (group -> genes) -----------------------------
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

# ----------------------------- parsing helpers -----------------------------
AA3 = {"Ala":"A","Arg":"R","Asn":"N","Asp":"D","Cys":"C","Gln":"Q","Glu":"E","Gly":"G","His":"H","Ile":"I",
       "Leu":"L","Lys":"K","Met":"M","Phe":"F","Pro":"P","Ser":"S","Thr":"T","Trp":"W","Tyr":"Y","Val":"V",
       "Ter":"*","Sec":"U"}
PROT_RE = re.compile(r"\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=|\*|\?)\)")
FS_RE   = re.compile(r"\(p\.[A-Z][a-z]{2}\d+[A-Za-z]{0,3}fs")
TX_RE   = re.compile(r"^[NX][MR]_\d+\.\d+")
CDNA_RE = re.compile(r":c\.([-*]?)(\d+)([+-]\d+)?")
STARS = {"practice guideline":4, "reviewed by expert panel":3,
         "criteria provided, multiple submitters, no conflicts":2,
         "criteria provided, single submitter":1,
         "criteria provided, conflicting classifications":1,
         "criteria provided, conflicting interpretations":1}

def stars(rs):
    return STARS.get(rs.strip().lower(), 0)

def classify(name):
    """-> (variant_class, ref_aa, alt_aa, aa_pos)"""
    m = PROT_RE.search(name)
    if m:
        ref3, pos, alt3 = m.groups()
        ref = AA3.get(ref3, "?"); pos = int(pos)
        if alt3 == "=":                          return "synonymous", ref, ref, pos
        if alt3 in ("Ter", "*"):                 return "nonsense", ref, "*", pos
        if alt3 == "?":                          return ("start_lost" if (ref3 == "Met" and pos == 1) else "protein_unknown"), ref, "?", pos
        if ref3 == "Met" and pos == 1:           return "start_lost", ref, AA3.get(alt3, "?"), pos   # p.Met1Val etc.
        return "missense", ref, AA3.get(alt3, "?"), pos
    if FS_RE.search(name):                       return "frameshift", ".", ".", 0
    c = CDNA_RE.search(name)
    if c:
        sign, _, off = c.groups()
        if sign == "-":                          return "utr5", ".", ".", 0
        if sign == "*":                          return "utr3", ".", ".", 0
        if off:
            d = abs(int(off))
            if d <= 2:                           return "splice_canonical", ".", ".", 0
            if d <= 8:                           return "splice_region", ".", ".", 0
            return "intronic", ".", ".", 0
        return "coding_no_protein_annot", ".", ".", 0
    return "other_noncoding", ".", ".", 0

def label_from_sig(sig):
    s = sig.lower()
    path, ben, conf, unc = "pathogenic" in s, "benign" in s, "conflict" in s, "uncertain" in s
    if path and not (conf or unc or ben): return 1
    if ben and not (conf or unc or path): return 0
    if unc and not conf and not path and not ben: return 2          # VUS
    return -1                                                        # conflicting / other

def panel_gene(symbols):
    hits = [g for g in symbols.split(";") if g in GENE2SYN]
    return hits[0] if len(hits) == 1 else (None if not hits else "AMBIGUOUS")

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gz"); ap.add_argument("--outdir", default="clinvar_out")
    ap.add_argument("--min-stars", type=int, default=1, help="min review stars for labelled benign/pathogenic")
    ap.add_argument("--indels", action="store_true", help="also write labelled small indels (<=50bp)")
    ap.add_argument("--hash", action="store_true", help="sha256 the input (slow on big files)")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    need = ["GeneSymbol","Assembly","OriginSimple","ClinicalSignificance","ReviewStatus","Type","Name",
            "Chromosome","PositionVCF","ReferenceAlleleVCF","AlternateAlleleVCF"]
    out_cols = ["variation_id","gene","syndrome_group","chrom","pos","ref","alt","label","clinical_significance",
                "review_status","stars","variant_class","ref_aa","alt_aa","aa_pos","transcript","hgvs_name","last_evaluated"]
    files = {"labelled": "clinvar_labelled_snv.csv", "vus": "clinvar_vus_snv.csv",
             "conflict": "clinvar_conflicting_snv.csv"}
    if a.indels: files["indel"] = "clinvar_labelled_indels.csv"
    fh = {k: open(os.path.join(a.outdir, v), "w", newline="", encoding="utf-8") for k, v in files.items()}
    wr = {k: csv.writer(f) for k, f in fh.items()}
    for w in wr.values(): w.writerow(out_cols)

    c = Counter(); per_gene = defaultdict(Counter); seen = {}; dropped_dupe_conflict = 0
    with gzip.open(a.gz, "rt", encoding="utf-8", errors="ignore") as f:
        header = f.readline().lstrip("#").rstrip("\n").split("\t")
        col = {n: i for i, n in enumerate(header)}
        miss = [n for n in need if n not in col]
        if miss: sys.exit(f"Missing columns in input: {miss}\nHeader: {header}")
        g = lambda p, n: p[col[n]] if n in col and col[n] < len(p) else ""
        for line in f:
            p = line.rstrip("\n").split("\t"); c["rows"] += 1
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
            if ref == alt: c["ref_equals_alt_skipped"] += 1; continue      # HGVS 'c.123=' entries
            snv = len(ref) == 1 and len(alt) == 1
            if not snv and not (a.indels and max(len(ref), len(alt)) <= 50): c["not_snv_skipped"] += 1; continue
            lab = label_from_sig(g(p, "ClinicalSignificance")); st = stars(g(p, "ReviewStatus"))
            if lab in (0, 1) and st < a.min_stars: c["dropped_low_stars"] += 1; continue
            if lab == -1 and not snv: continue
            key = (g(p, "Chromosome"), g(p, "PositionVCF"), ref, alt, gene)
            if key in seen:
                c["duplicate_rows"] += 1
                if seen[key] != lab: dropped_dupe_conflict += 1
                continue
            seen[key] = lab
            vc, raa, aaa, apos = classify(g(p, "Name"))
            row = [g(p, "VariationID"), gene, GENE2SYN[gene], key[0], key[1], ref, alt, lab,
                   g(p, "ClinicalSignificance"), g(p, "ReviewStatus"), st, vc, raa, aaa, apos,
                   (TX_RE.search(g(p, "Name")) or [""])[0] if TX_RE.search(g(p, "Name")) else "", g(p, "Name"), g(p, "LastEvaluated")]
            if not snv:
                if lab in (0, 1): wr["indel"].writerow(row); c["indel_labelled"] += 1
                continue
            dest = {0: "labelled", 1: "labelled", 2: "vus", -1: "conflict"}[lab]
            wr[dest].writerow(row)
            tag = {0: "benign", 1: "pathogenic", 2: "vus", -1: "conflict"}[lab]
            c[f"snv_{tag}"] += 1; per_gene[gene][tag] += 1
            if lab in (0, 1): per_gene[gene][f"cls_{vc}"] += 1
    for f_ in fh.values(): f_.close()

    # ---- post-pass: flag rows on each gene's most common transcript (aa_pos only comparable within one transcript) ----
    tx = defaultdict(Counter)
    for k in ("labelled", "vus"):
        with open(os.path.join(a.outdir, files[k]), newline="", encoding="utf-8") as fi:
            for r in csv.DictReader(fi):
                if r["transcript"]: tx[r["gene"]][r["transcript"]] += 1
    primary = {gn: cnt.most_common(1)[0][0] for gn, cnt in tx.items()}
    off_primary = 0
    for k, fn in files.items():
        path = os.path.join(a.outdir, fn); tmp = path + ".tmp"
        with open(path, newline="", encoding="utf-8") as fi, open(tmp, "w", newline="", encoding="utf-8") as fo:
            rd = csv.reader(fi); wo = csv.writer(fo); hdr = next(rd); wo.writerow(hdr + ["on_primary_transcript"])
            ig, it = hdr.index("gene"), hdr.index("transcript")
            for r in rd:
                flag = int(bool(r[it]) and r[it] == primary.get(r[ig]))
                if k == "labelled" and not flag: off_primary += 1
                wo.writerow(r + [flag])
        os.replace(tmp, path)
    c["labelled_snv_off_primary_transcript"] = off_primary

    # ---- manifest + printed report ----
    st_ = os.stat(a.gz)
    gene_table = {gn: {"group": GENE2SYN[gn], **{k: per_gene[gn][k] for k in ("benign","pathogenic","vus","conflict")}}
                  for gn in GENE2SYN}
    zero = [gn for gn, v in gene_table.items() if v["benign"] + v["pathogenic"] == 0]
    sparse = [gn for gn, v in gene_table.items() if 0 < v["benign"] + v["pathogenic"] < 30]
    man = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "input": {"file": os.path.basename(a.gz),
           "bytes": st_.st_size, "mtime": time.strftime("%Y-%m-%d", time.localtime(st_.st_mtime))},
           "params": {"min_stars": a.min_stars, "indels": a.indels}, "n_panel_genes": len(GENE2SYN),
           "counts": dict(c), "primary_transcripts": primary, "dropped_duplicate_label_conflicts": dropped_dupe_conflict,
           "genes_with_zero_labelled": zero, "genes_with_lt30_labelled": sparse, "per_gene": gene_table}
    if a.hash:
        h = hashlib.sha256()
        with open(a.gz, "rb") as fb:
            for blk in iter(lambda: fb.read(1 << 24), b""): h.update(blk)
        man["input"]["sha256"] = h.hexdigest()
    json.dump(man, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=2)

    print(f"Panel genes: {len(GENE2SYN)} in {len(PANEL)} groups")
    for k in ["rows","panel_rows","grch38","germline","snv_benign","snv_pathogenic","snv_vus","snv_conflict",
              "dropped_low_stars","duplicate_rows","skipped_overlapping_panel_genes","not_snv_skipped","ref_equals_alt_skipped","labelled_snv_off_primary_transcript","indel_labelled"]:
        print(f"  {k:34s}{c[k]:>10,}")
    print(f"Genes with 0 labelled variants: {zero or 'none'}")
    print(f"Genes with <30 labelled variants (too sparse to evaluate): {sparse or 'none'}")
    print(f"Wrote files to: {a.outdir}")

if __name__ == "__main__":
    main()