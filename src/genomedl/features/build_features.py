#!/usr/bin/env python3
"""
build_features.py — GenomeDL v3 Feature Engineering Pipeline
=============================================================
Applies all Stage 0 feature fixes on top of v2:

  0.3 Protein lengths from data/reference/panel.json (MANE/UniProt); warns on fallback.
  0.4 Coding-strand nucleotide one-hots (cs_ref_A/C/G/T, cs_alt_A/C/G/T)
      using strand from data/reference/panel.json.
  0.5 start_lost rows get fixed severity (grantham=215, tanimoto=1.0,
      delta_mw = -Met_mw) instead of actual Met->X Grantham value.
  0.6 All gene-level metadata read from panel.json (no hardcoded fallbacks).
  0.7 Unknown gene raises ValueError immediately.
  Also adds stop_lost (class 12) support throughout.

Output: data/processed/v3_featured_snv.csv (all v2 columns + cs_* one-hots)

Usage (from repo root):
    python -m genomedl.features.build_features \\
        --input  data/processed/clinvar_labelled_v3.csv \\
        --panel  data/reference/panel.json \\
        --output data/processed/v3_featured_snv.csv
"""
import argparse, json, math, os, sys, time, warnings
import numpy as np
import pandas as pd

from genomedl.paths import CLINVAR_LABELLED_CSV, FEATURED_SNV_CSV, PANEL_JSON

warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, DataStructs
RDLogger.DisableLog("rdApp.*")

# ─────────────────────── RDKit amino-acid chemistry ─────────────────────────
AA_SMILES = {
    "A":"CC(N)C(=O)O","R":"NC(CCCNC(=N)N)C(=O)O","N":"NC(CC(=O)N)C(=O)O",
    "D":"NC(CC(=O)O)C(=O)O","C":"NC(CS)C(=O)O","Q":"NC(CCC(=O)N)C(=O)O",
    "E":"NC(CCC(=O)O)C(=O)O","G":"NCC(=O)O","H":"NC(Cc1c[nH]cn1)C(=O)O",
    "I":"CCC(C)C(N)C(=O)O","L":"CC(C)CC(N)C(=O)O","K":"NCCCCC(N)C(=O)O",
    "M":"CSCCC(N)C(=O)O","F":"NC(Cc1ccccc1)C(=O)O","P":"O=C(O)C1CCCN1",
    "S":"NC(CO)C(=O)O","T":"CC(O)C(N)C(=O)O","W":"NC(Cc1c[nH]c2ccccc12)C(=O)O",
    "Y":"NC(Cc1ccc(O)cc1)C(=O)O","V":"CC(C)C(N)C(=O)O",
}
AA_CHARGE = {
    "R":1.0,"K":1.0,"H":0.1,"D":-1.0,"E":-1.0,
    "A":0.0,"N":0.0,"C":0.0,"Q":0.0,"G":0.0,"I":0.0,"L":0.0,
    "M":0.0,"F":0.0,"P":0.0,"S":0.0,"T":0.0,"W":0.0,"Y":0.0,"V":0.0,
}
def _build_aa_props():
    props = {}
    for code, smi in AA_SMILES.items():
        mol = Chem.MolFromSmiles(smi)
        fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        props[code] = {
            "mw": float(Descriptors.ExactMolWt(mol)),
            "logp": float(Descriptors.MolLogP(mol)),
            "tpsa": float(Descriptors.TPSA(mol)),
            "charge": AA_CHARGE[code], "fp": fp,
        }
    return props
AA_PROPS = _build_aa_props()
MET_MW   = AA_PROPS["M"]["mw"]   # used for start_lost delta_mw

# ─────────────────────── Grantham (1974) matrix ──────────────────────────────
GRANTHAM_CONSTS = {
    "S":(1.42,9.2,32.0),"R":(0.65,10.5,124.0),"L":(0.0,4.9,111.0),
    "P":(0.39,8.0,32.5),"T":(0.71,8.6,61.0),"A":(0.0,8.1,31.0),
    "V":(0.0,5.9,84.0),"G":(0.74,9.0,3.0),"I":(0.0,5.2,111.0),
    "F":(0.0,5.2,132.0),"Y":(0.20,6.2,136.0),"C":(2.75,5.5,55.0),
    "H":(0.58,10.4,96.0),"Q":(0.89,10.5,85.0),"N":(1.33,11.6,56.0),
    "K":(0.33,11.3,119.0),"D":(1.38,13.0,54.0),"E":(0.92,12.3,83.0),
    "M":(0.0,5.7,105.0),"W":(0.13,5.4,170.0),
}
def _build_grantham():
    mat = {}
    for a1,(c1,p1,v1) in GRANTHAM_CONSTS.items():
        for a2,(c2,p2,v2) in GRANTHAM_CONSTS.items():
            if a1==a2: mat[(a1,a2)]=0.0
            else: mat[(a1,a2)]=round(50.723*math.sqrt(
                1.833*(c1-c2)**2+0.1018*(p1-p2)**2+0.000399*(v1-v2)**2),1)
    return mat
GRANTHAM = _build_grantham()

# ─────────────────────── variant classes (12 total) ──────────────────────────
VARIANT_CLASSES = [
    "synonymous","missense","nonsense","start_lost","stop_lost",
    "splice_canonical","splice_region","intronic",
    "utr5","utr3","other_noncoding","coding_no_protein_annot",
]
CLASS_TO_ID = {c:i for i,c in enumerate(VARIANT_CLASSES)}

TRANSITIONS = {("A","G"),("G","A"),("C","T"),("T","C")}
COMP = str.maketrans("ACGT","TGCA")

# ─────────────────────── main pipeline ──────────────────────────────────────
def featurize(input_csv: str, panel_path: str, output_csv: str) -> None:
    t0 = time.time()
    print("=" * 70)
    print("GenomeDL v3 — Feature Engineering Pipeline (Stage 0)")
    print("=" * 70)

    # Load panel.json (single source of truth)
    if not os.path.exists(panel_path):
        sys.exit(f"ERROR: panel.json not found at {panel_path}. "
                 f"Run make_panel_json.py first.")
    panel = json.load(open(panel_path))
    print(f"Loaded panel.json: {len(panel)} genes")

    # Load dataset
    print(f"Loading {input_csv} ...")
    df = pd.read_csv(input_csv)
    n  = len(df)
    print(f"  Rows: {n:,}  |  Genes in CSV: {df['gene'].nunique()}")

    # [0.7] Unknown gene check — fail fast
    csv_genes = set(df["gene"].unique())
    panel_genes = set(panel.keys())
    unknown = csv_genes - panel_genes
    if unknown:
        raise ValueError(
            f"[0.7] Gene(s) not in 130-gene panel: {sorted(unknown)}. "
            "Remove these rows or update the panel before proceeding."
        )

    # [0.3] Protein lengths from panel.json (not max aa_pos)
    gene_lengths = {g: panel[g]["protein_length"] for g in panel}
    missing = [g for g in csv_genes if gene_lengths.get(g, 0) == 0]
    if missing:
        print(f"  WARNING [0.3]: {len(missing)} genes have length=0 in panel.json "
              f"(fallback to max aa_pos): {missing}")
        for g in missing:
            gene_lengths[g] = max(int(df[df["gene"]==g]["aa_pos"].max()), 100)

    # [0.4] Strand map from panel.json
    strand_map = {g: panel[g]["strand"] for g in panel}

    # Categorical encodings (sorted for reproducibility)
    unique_genes     = sorted(csv_genes)
    unique_syndromes = sorted(df["syndrome_group"].unique())
    gene_to_id  = {g: i for i, g in enumerate(unique_genes)}
    syn_to_id   = {s: i for i, s in enumerate(unique_syndromes)}
    print(f"  Genes: {len(unique_genes)}  |  Syndromes: {len(unique_syndromes)}")

    # Pre-allocate arrays
    has_protein  = np.zeros(n, np.int32)
    rel_pos      = np.zeros(n, np.float32)
    grantham     = np.zeros(n, np.float32)
    d_mw         = np.zeros(n, np.float32)
    d_charge     = np.zeros(n, np.float32)
    d_logp       = np.zeros(n, np.float32)
    d_tpsa       = np.zeros(n, np.float32)
    tanimoto     = np.zeros(n, np.float32)
    is_trans     = np.zeros(n, np.int32)
    gene_ids     = np.zeros(n, np.int32)
    syn_ids      = np.zeros(n, np.int32)
    vcls_ids     = np.zeros(n, np.int32)
    # [0.4] coding-strand one-hots (8 columns)
    cs_ref = {b: np.zeros(n, np.int32) for b in "ACGT"}
    cs_alt = {b: np.zeros(n, np.int32) for b in "ACGT"}

    ra_col    = df["ref_aa"].values
    aa_col    = df["alt_aa"].values
    pos_col   = df["aa_pos"].values
    gene_col  = df["gene"].values
    syn_col   = df["syndrome_group"].values
    vcls_col  = df["variant_class"].values
    ref_col   = df["ref"].values
    alt_col   = df["alt"].values

    print(f"Processing {n:,} rows ...")
    for i in range(n):
        gene  = gene_col[i]
        vcls  = vcls_col[i]
        ra    = ra_col[i]
        aa    = aa_col[i]
        pos   = pos_col[i]
        g_len = gene_lengths.get(gene, 1000)
        strand = strand_map.get(gene, "+")

        gene_ids[i] = gene_to_id[gene]
        syn_ids[i]  = syn_to_id.get(syn_col[i], 0)
        vcls_ids[i] = CLASS_TO_ID.get(vcls, CLASS_TO_ID["other_noncoding"])

        # Transition / Transversion
        r_base = str(ref_col[i]).upper()
        a_base = str(alt_col[i]).upper()
        is_trans[i] = 1 if (r_base, a_base) in TRANSITIONS else 0

        # [0.4] Coding-strand one-hots
        # For minus-strand genes, the coding-strand base is the complement
        if strand == "-":
            cs_r = r_base.translate(COMP) if r_base in "ACGT" else r_base
            cs_a = a_base.translate(COMP) if a_base in "ACGT" else a_base
        else:
            cs_r, cs_a = r_base, a_base
        for b in "ACGT":
            cs_ref[b][i] = 1 if cs_r == b else 0
            cs_alt[b][i] = 1 if cs_a == b else 0

        is_std_ra = ra in AA_PROPS
        is_std_aa = aa in AA_PROPS

        # [0.5] start_lost: fixed maximum severity (NOT actual Met->X Grantham)
        if vcls == "start_lost":
            has_protein[i]  = 1
            rel_pos[i]      = 0.0           # always position 1
            grantham[i]     = 215.0         # fixed max severity (> any real pair)
            tanimoto[i]     = 1.0           # fully unlike
            d_mw[i]         = -MET_MW       # loss of Met
            d_charge[i]     = 0.0
            d_logp[i]       = 0.0
            d_tpsa[i]       = 0.0

        elif vcls == "stop_lost":
            # Readthrough: ref=*, alt=some AA
            has_protein[i] = 1
            rel_pos[i]     = min(1.0, max(0.0, pos / g_len)) if pos > 0 else 1.0
            grantham[i]    = 200.0          # severe (protein gains extra AA)
            tanimoto[i]    = 1.0
            if is_std_aa:
                p_alt = AA_PROPS[aa]
                d_mw[i]     = p_alt["mw"]  # gain of an amino acid
                d_charge[i] = p_alt["charge"]
                d_logp[i]   = p_alt["logp"]
                d_tpsa[i]   = p_alt["tpsa"]

        elif is_std_ra and is_std_aa:
            has_protein[i] = 1
            rel_pos[i]     = min(1.0, max(0.0, pos / g_len)) if pos > 0 else 0.0
            grantham[i]    = GRANTHAM.get((ra, aa), 100.0)
            p_ref, p_alt   = AA_PROPS[ra], AA_PROPS[aa]
            d_mw[i]        = round(p_alt["mw"]   - p_ref["mw"],   3)
            d_charge[i]    = round(p_alt["charge"]- p_ref["charge"],3)
            d_logp[i]      = round(p_alt["logp"]  - p_ref["logp"], 3)
            d_tpsa[i]      = round(p_alt["tpsa"]  - p_ref["tpsa"], 3)
            tanimoto[i]    = 0.0 if ra == aa else round(
                1.0 - DataStructs.TanimotoSimilarity(p_ref["fp"], p_alt["fp"]), 4)

        elif aa == "*" or ra == "*" or vcls == "nonsense":
            has_protein[i] = 1
            rel_pos[i]     = min(1.0, max(0.0, pos / g_len)) if pos > 0 else 0.0
            grantham[i]    = 200.0
            tanimoto[i]    = 1.0
            if is_std_ra:
                p_ref = AA_PROPS[ra]
                d_mw[i]     = -p_ref["mw"]
                d_charge[i] = -p_ref["charge"]
                d_logp[i]   = -p_ref["logp"]
                d_tpsa[i]   = -p_ref["tpsa"]
        # else: non-coding / intronic / UTR — all zeros (default)

    # Assemble output DataFrame
    df["has_protein_change"]  = has_protein
    df["rel_protein_pos"]     = rel_pos
    df["grantham_score"]      = grantham
    df["delta_mw"]            = d_mw
    df["delta_charge"]        = d_charge
    df["delta_logp"]          = d_logp
    df["delta_tpsa"]          = d_tpsa
    df["tanimoto_chem_dist"]  = tanimoto
    df["is_transition"]       = is_trans
    df["gene_id"]             = gene_ids
    df["syndrome_id"]         = syn_ids
    df["variant_class_id"]    = vcls_ids

    # Plus-strand one-hots (kept for v2 compatibility)
    for b in "ACGT":
        df[f"ref_{b}"] = (df["ref"] == b).astype(np.int32)
        df[f"alt_{b}"] = (df["alt"] == b).astype(np.int32)

    # [0.4] Coding-strand one-hots (new in v3)
    for b in "ACGT":
        df[f"cs_ref_{b}"] = cs_ref[b]
        df[f"cs_alt_{b}"] = cs_alt[b]

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)

    elapsed = time.time() - t0
    mb = os.path.getsize(output_csv) / 1e6
    print("\n" + "=" * 70)
    print(f"DONE in {elapsed:.1f}s  |  {output_csv}  ({mb:.1f} MB)")
    print(f"Rows: {n:,}  |  Columns: {df.shape[1]}")
    print("\nVariant class distribution:")
    for vc in VARIANT_CLASSES:
        sub = df[df["variant_class"] == vc]
        if len(sub) > 0:
            lbl1 = (sub["label"] == 1).sum()
            print(f"  {vc:<30} {len(sub):>8,}  path={lbl1:,} ({100*lbl1/len(sub):.1f}%)")
    print("\nValidation checks:")
    # Cys->Gly Grantham
    cg = GRANTHAM.get(("C","G"), None)
    print(f"  Cys->Gly Grantham : {cg}  (expect ~158-159)")
    # start_lost severity
    sl = df[df["variant_class"]=="start_lost"]
    if len(sl) > 0:
        print(f"  start_lost grantham: {sl['grantham_score'].unique()} (expect [215.])")
    print("=" * 70)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input",  default=str(CLINVAR_LABELLED_CSV))
    ap.add_argument("--panel",  default=str(PANEL_JSON))
    ap.add_argument("--output", default=str(FEATURED_SNV_CSV))
    a = ap.parse_args()
    featurize(a.input, a.panel, a.output)
