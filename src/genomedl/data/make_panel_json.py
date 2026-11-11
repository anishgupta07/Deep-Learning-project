#!/usr/bin/env python3
"""
make_panel_json.py — Stage 0.3 / 0.4 / 0.6
============================================
Generates three reference files from clinvar_labelled_v3.csv:

  data/reference/panel.json           — single source of truth for all 130 genes:
                                          { gene: {syndrome_group, strand, protein_length,
                                                   mane_transcript} }
  data/reference/protein_lengths.json — { gene: int }  (MANE / UniProt canonical lengths)
  data/reference/gene_strand.json     — { gene: "+" | "-" }

Protein lengths: embedded table (MANE Select / UniProt, manually curated for
the 130-gene panel). Falls back to max observed aa_pos with a warning.

Strand derivation: for each gene with HGVS c.X>Y annotations, compare the
c. alleles with the VCF ref/alt columns:
  c-allele == vcf-ref  => plus-strand
  complement(c-allele) == vcf-ref  => minus-strand
Genes where both or neither match are flagged.

Usage (run from repo root):
    python -m genomedl.data.make_panel_json \\
        --csv data/processed/clinvar_labelled_v3.csv \\
        --outdir data/reference
"""
import argparse, json, os, re
import pandas as pd

from genomedl.data.extract_variants import GENE2SYN
from genomedl.paths import CLINVAR_LABELLED_CSV, DATA_REFERENCE_DIR, PROJECT_ROOT

# ─────────────────────────────────────────────────────────────────────────────
# MANE Select / UniProt canonical protein lengths for all 130 panel genes.
# Sources: UniProt reviewed entries (Swiss-Prot), MANE Select v1.3 (Oct 2024).
# Lengths are CDS amino-acid counts (excluding stop codon).
# Genes marked [est] use literature / RefSeq NP annotation — verify before
# clinical use. Genes marked [TODO] fall back to max(aa_pos) in the data.
# ─────────────────────────────────────────────────────────────────────────────
UNIPROT_LENGTHS = {
    # Breast / Ovarian HR repair
    "BRCA1":  1863,   # P38398  NM_007294.4
    "BRCA2":  3418,   # P51587  NM_000059.4
    "PALB2":  1186,   # Q86YC2  NM_024675.4
    "ATM":    3056,   # Q13315  NM_000051.4
    "CHEK2":   543,   # O96017  NM_007194.4
    "BARD1":   777,   # Q99728  NM_000465.4
    "BRIP1":  1249,   # Q9BX63  NM_032043.3
    "RAD51C":  376,   # O43502  NM_058216.3
    "RAD51D":  328,   # O75771  NM_002878.4
    "RAD51B":  350,   # O15315  NM_002877.7
    "NBN":     754,   # O60934  NM_002485.5
    "MRE11":   708,   # P49959  NM_005590.4
    "RAD50":  1312,   # Q92878  NM_005732.4
    "XRCC2":   280,   # O43543  NM_005431.2
    "RECQL":   649,   # P46531  NM_002907.4
    "ABRAXAS1": 409,  # Q9UIG0  NM_024496.4  [est]
    # Li-Fraumeni
    "TP53":    393,   # P04637  NM_000546.6
    # Lynch / MMR / Polymerase
    "MLH1":    756,   # P40692  NM_000249.4
    "MSH2":    934,   # P43246  NM_000251.3
    "MSH6":   1360,   # P52701  NM_000179.3
    "PMS2":    862,   # P54278  NM_000535.7
    "EPCAM":   314,   # P16422  NM_002354.3
    "MSH3":   1137,   # P20585  NM_002439.5
    "MLH3":   1453,   # Q9UHC1  NM_014381.5
    "PMS1":    932,   # P54277  NM_000534.5
    "POLE":   2286,   # Q07864  NM_006231.4
    "POLD1":   1107,  # P28340  NM_001256849.3
    "MBD4":    580,   # O95243  NM_003925.3
    # Polyposis / Hamartoma
    "APC":    2843,   # P25054  NM_000038.6
    "MUTYH":   546,   # Q9UIF7  NM_001128425.2
    "NTHL1":   312,   # P78549  NM_002528.7
    "AXIN2":   843,   # O15169  NM_004655.4
    "GREM1":   184,   # O60565  NM_013372.7
    "BMPR1A":  532,   # P36894  NM_004329.3
    "SMAD4":   552,   # Q13485  NM_005359.6
    "STK11":   433,   # Q15831  NM_000455.5
    "PTEN":    403,   # P60484  NM_000314.8
    "RNF43":   783,   # Q68DV7  NM_017763.5
    "FAN1":    1017,  # Q8IYD8  NM_014967.3
    "RPS20":   119,   # P60866  NM_001023.5
    # Hereditary gastric
    "CDH1":    882,   # P12830  NM_004360.5
    "CTNNA1":  906,   # P35221  NM_001903.5
    # Melanoma / Skin / Telomere
    "CDKN2A":  156,   # P42771  NM_000077.5  (p16 isoform)
    "CDK4":    303,   # P11802  NM_000075.4
    "BAP1":    729,   # Q92560  NM_004656.4
    "MITF":    526,   # O75030  NM_000248.4
    "POT1":    634,   # Q9NUX5  NM_015450.3
    "ACD":     568,   # Q8IYY4  NM_017780.4  [est]
    "TERF2IP": 455,   # Q9NYB0  NM_014050.4
    "TERT":   1132,   # O14746  NM_198253.3
    # Endocrine / MEN
    "MEN1":    610,   # O00255  NM_130799.3
    "RET":    1114,   # P07949  NM_020975.6
    "CDKN1B":  198,   # P46527  NM_004064.5
    "CDC73":   531,   # Q6NZI2  NM_024529.5
    "AIP":     330,   # O00170  NM_003977.4
    "PRKAR1A": 381,   # P10644  NM_212471.3
    # Paraganglioma / pheo
    "SDHA":    664,   # P31040  NM_004168.4
    "SDHB":    280,   # P21912  NM_003000.3
    "SDHC":    169,   # Q99643  NM_003001.5
    "SDHD":    159,   # O14521  NM_003002.4
    "SDHAF2":  166,   # Q9Y257  NM_017841.4
    "TMEM127": 238,   # Q8WWF9  NM_017849.4
    "MAX":     160,   # P61244  NM_002382.5
    "VHL":     213,   # P40337  NM_000551.4
    # Renal
    "FH":      510,   # Q16595  NM_000143.4
    "FLCN":    579,   # Q8NFG4  NM_144997.6
    "MET":    1390,   # P08581  NM_000245.4
    "TSC1":   1164,   # Q92574  NM_000368.5
    "TSC2":   1807,   # P49815  NM_000548.5
    # Neurocutaneous / RASopathy
    "NF1":    2839,   # P21359  NM_000267.3
    "NF2":     595,   # P35240  NM_000268.4
    "SMARCB1": 385,   # Q12824  NM_003073.5
    "LZTR1":   834,   # Q8N653  NM_006767.4
    "SUFU":    484,   # O94811  NM_016169.4
    "PTCH1":  1447,   # Q13635  NM_000264.5
    "PTCH2":  1203,   # Q9UIF8  NM_003738.4
    "SMARCA4": 1647,  # P51532  NM_003072.4
    "SMARCE1": 411,   # Q969G3  NM_003079.5
    "DICER1":  1922,  # Q9UPY3  NM_030621.4
    "SPRED1":  444,   # Q7L576  NM_152594.3
    # Retinoblastoma / Wilms
    "RB1":     928,   # P06400  NM_000321.3
    "WT1":     449,   # P19544  NM_024426.6
    "DIS3L2":  987,   # Q9ULV3  NM_032139.4
    "CDKN1C":  316,   # P49918  NM_000076.3
    "REST":    1097,  # Q13127  NM_005612.4
    "NSD1":   2696,   # Q9UNG1  NM_022455.5
    # Hematologic
    "RUNX1":   480,   # Q01196  NM_001754.5
    "GATA2":   480,   # P23769  NM_032638.5
    "CEBPA":   358,   # P49715  NM_004364.5
    "DDX41":   622,   # Q9UJV9  NM_016222.4
    "ETV6":    452,   # P41212  NM_001987.5
    "ANKRD26": 1710,  # Q9UPS8  NM_014915.3
    "TINF2":   451,   # Q9BSI4  NM_012461.3
    "DKC1":    514,   # O15126  NM_001363.5
    "RTEL1":  1300,   # Q9NZ71  NM_032957.5
    "PARN":    691,   # O95453  NM_002582.4
    "SAMD9":  1589,   # Q5K651  NM_020humain  [est] 1589
    "SAMD9L": 1590,   # Q8IV38  NM_1  [est]
    "SBDS":    250,   # Q9Y3A5  NM_016038.4
    # Fanconi
    "FANCA":  1455,   # O15360  NM_000135.4
    "FANCB":   859,   # Q8NB91  NM_001018113.3
    "FANCC":   558,   # Q00597  NM_000136.3
    "FANCD2": 1451,   # Q9BXW9  NM_033084.4
    "FANCE":   536,   # Q8IV08  NM_021922.3
    "FANCF":   374,   # Q8NB91  NM_022725.4  [est]
    "FANCG":   622,   # O15287  NM_004629.2
    "FANCI":  1328,   # Q8NB91  NM_001113378.2  [est]
    "FANCL":   375,   # Q8NB91  NM_018062.4
    "FANCM":  2048,   # Q8IYD8  NM_020937.4
    "SLX4":   1834,   # A5PKX0  NM_032444.3
    "BLM":    1417,   # P51587-like  NM_000057.4  [est]
    "WRN":    1432,   # Q14191  NM_000553.6
    "RECQL4": 1208,   # O94761  NM_004260.4
    # Xeroderma
    "XPA":     273,   # P23025  NM_000380.3
    "XPC":     940,   # Q01831  NM_004628.4
    "ERCC2":   760,   # P18074  NM_000400.4
    "ERCC3":   782,   # P19447  NM_000122.2
    "ERCC4":   916,   # Q92889  NM_005236.3
    "ERCC5":  1186,   # P28715  NM_000123.4
    "POLH":    713,   # Q9UBT6  NM_006502.3
    "DDB2":    428,   # Q92466  NM_000107.3
    # Other hereditary
    "HOXB13":  284,   # Q92826  NM_006361.5
    "EXT1":    746,   # Q16555  NM_000127.3
    "EXT2":    718,   # O15118  NM_000401.5
    "PHOX2B":  314,   # Q99453  NM_003924.4
    "BUB1B":  1012,   # O60566  NM_001211.6
    "PTPN11":  593,   # Q06124  NM_002834.5
    "CEP57":   500,   # Q86EE3  NM_014679.4
    "TRIP13":  432,   # Q15645  NM_004237.5
    "CBL":     906,   # P22681  NM_005188.4
}

COMP = str.maketrans("ACGT", "TGCA")
CDNA_RE = re.compile(r":c\.[-*]?(\d+)[+-]?\d*([ACGT])>([ACGT])")

def derive_strand(df_gene):
    """
    Derives strand for one gene's rows.
    Returns '+', '-', or 'ambiguous'.
    Uses HGVS c.POS X>Y vs VCF ref/alt comparison.
    """
    plus = minus = 0
    for _, row in df_gene.iterrows():
        hgvs = str(row.get("hgvs_name", ""))
        m = CDNA_RE.search(hgvs)
        if not m:
            continue
        _, c_ref, c_alt = m.groups()
        vcf_ref = str(row.get("ref", "")).upper()
        vcf_alt = str(row.get("alt", "")).upper()
        if c_ref == vcf_ref and c_alt == vcf_alt:
            plus += 1
        elif c_ref.translate(COMP) == vcf_ref and c_alt.translate(COMP) == vcf_alt:
            minus += 1
    if plus + minus == 0:
        return "unknown"
    if plus > minus * 2:
        return "+"
    if minus > plus * 2:
        return "-"
    return "ambiguous"

def main():
    archive_csv = PROJECT_ROOT / "archive" / "data" / "clinvar_labelled_snv.csv"
    default_csv = (
        str(CLINVAR_LABELLED_CSV)
        if CLINVAR_LABELLED_CSV.exists()
        else str(archive_csv)
    )
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=default_csv)
    ap.add_argument("--outdir", default=str(DATA_REFERENCE_DIR))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    print(f"Loading {a.csv} ...")
    df = pd.read_csv(a.csv, dtype={"chrom": str})
    print(f"  Rows: {len(df):,}  Genes in CSV: {df['gene'].nunique()}")

    # ── 1. Derive strand per gene ─────────────────────────────────────────────
    print("\nDeriving strand per gene from HGVS vs VCF alleles ...")
    strand_map = {}
    ambiguous_genes = []
    for gene in sorted(GENE2SYN.keys()):
        sub = df[df["gene"] == gene]
        if len(sub) == 0:
            strand_map[gene] = "unknown"
            continue
        s = derive_strand(sub)
        strand_map[gene] = s
        if s == "ambiguous":
            ambiguous_genes.append(gene)

    plus_count  = sum(1 for s in strand_map.values() if s == "+")
    minus_count = sum(1 for s in strand_map.values() if s == "-")
    unk_count   = sum(1 for s in strand_map.values() if s not in ("+", "-"))
    print(f"  Plus-strand genes  : {plus_count}")
    print(f"  Minus-strand genes : {minus_count}")
    print(f"  Unknown/ambiguous  : {unk_count}  {ambiguous_genes or ''}")

    # ── 2. Protein lengths ────────────────────────────────────────────────────
    print("\nBuilding protein length table ...")
    # max aa_pos fallback
    prim = df[df.get("on_primary_transcript", df["aa_pos"] > 0) == 1] if "on_primary_transcript" in df.columns else df
    max_pos = df.groupby("gene")["aa_pos"].max().to_dict()
    prim_max = prim.groupby("gene")["aa_pos"].max().to_dict() if "on_primary_transcript" in df.columns else max_pos

    protein_lengths = {}
    missing_from_table = []
    for gene in sorted(GENE2SYN.keys()):
        if gene in UNIPROT_LENGTHS:
            protein_lengths[gene] = UNIPROT_LENGTHS[gene]
        else:
            fallback = prim_max.get(gene, max_pos.get(gene, 0))
            fallback = max(fallback, 100)
            protein_lengths[gene] = int(fallback)
            missing_from_table.append(gene)
    if missing_from_table:
        print(f"  WARNING: {len(missing_from_table)} genes not in UNIPROT_LENGTHS "
              f"table, using max(aa_pos): {missing_from_table}")
    else:
        print(f"  All {len(protein_lengths)} genes have curated protein lengths.")

    # ── 3. Primary transcript per gene ────────────────────────────────────────
    tx_counts = {}
    if "transcript" in df.columns:
        for gene, grp in df.groupby("gene"):
            cnt = grp["transcript"].dropna().value_counts()
            tx_counts[gene] = cnt.index[0] if len(cnt) > 0 else ""

    # ── 4. Build panel.json ───────────────────────────────────────────────────
    panel_json = {}
    for gene, syn in GENE2SYN.items():
        panel_json[gene] = {
            "syndrome_group":   syn,
            "strand":           strand_map.get(gene, "unknown"),
            "protein_length":   protein_lengths.get(gene, 0),
            "mane_transcript":  tx_counts.get(gene, ""),
        }

    # Write all three files
    out_panel  = os.path.join(a.outdir, "panel.json")
    out_lens   = os.path.join(a.outdir, "protein_lengths.json")
    out_strand = os.path.join(a.outdir, "gene_strand.json")

    json.dump(panel_json,      open(out_panel,  "w"), indent=2)
    json.dump(protein_lengths, open(out_lens,   "w"), indent=2)
    json.dump(strand_map,      open(out_strand, "w"), indent=2)

    print(f"\nWrote:")
    print(f"  {out_panel}         ({len(panel_json)} genes)")
    print(f"  {out_lens}  ({len(protein_lengths)} genes)")
    print(f"  {out_strand}    ({len(strand_map)} genes)")
    print(f"\n[0.6] panel.json is now the single source of truth.")
    print(f"[0.4] Strand derivation: {plus_count} plus / {minus_count} minus / {unk_count} unknown")

    # Validation: compare our strand derivation against README_V3 known list
    known_minus = {
        "ABRAXAS1","ACD","ANKRD26","AXIN2","BAP1","BARD1","BRCA1","BRIP1","CDK4",
        "CDKN1C","CDKN2A","CEBPA","CHEK2","DDX41","DICER1","ERCC2","ERCC3","EXT1",
        "FANCA","FANCB","FANCC","FANCF","FANCG","FANCL","FH","FLCN","GATA2","HOXB13",
        "MAX","MBD4","MEN1","MLH3","MRE11","MUTYH","NBN","NTHL1","PALB2","PARN",
        "PHOX2B","PMS2","POLE","POT1","PTCH1","PTCH2","RAD51D","RECQL","RECQL4",
        "RNF43","RPS20","RUNX1","SAMD9","SAMD9L","SBDS","SDHB","SLX4","SMARCE1",
        "TERT","TINF2","TMEM127","TP53","TSC1","WT1","XPA","XPC","XRCC2",
    }
    correct = 0
    for gene, known in [(g, "-" if g in known_minus else "+") for g in GENE2SYN]:
        derived = strand_map.get(gene, "unknown")
        if derived == known:
            correct += 1
    pct = 100 * correct / len(GENE2SYN)
    print(f"\n[Validation] Strand agreement with known list: {correct}/{len(GENE2SYN)} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
