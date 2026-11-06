"""
predict.py — GenomeDL v2 Inference + Cancer Risk Report
=========================================================
Given a genetic variant in one of 130 hereditary-cancer genes:
  1. Computes the exact biophysical/RDKit/Grantham features matching build_v2_features.py
  2. Runs GenomeDLResNetV2 to predict ClinVar pathogenicity probability: P(pathogenic)
  3. Predicts organ-specific lifetime cancer risk:
       Cancer Risk (%) = P(pathogenic) * Gene-Specific Penetrance (%)
     and compares directly against population baseline prevalence!

Supported inputs:
  - 1-letter or 3-letter amino acid codes (C / Cys, G / Gly)
  - Full demo mode across 4 diverse cancer pathways
"""

import argparse
import sys
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

# ─────────────────────────────────────────────────────────────
# 1. CANCER PENETRANCE TABLE (Lifetime risk % if Pathogenic)
#    Source: NCCN Guidelines, ACMG/ClinGen, published literature
# ─────────────────────────────────────────────────────────────
PENETRANCE = {
    # Breast / Ovarian / HR repair
    "BRCA1":  {"Breast Cancer": 72, "Ovarian Cancer": 44, "Pancreatic Cancer": 3, "Prostate Cancer": 8, "Colorectal Cancer": 2},
    "BRCA2":  {"Breast Cancer": 69, "Ovarian Cancer": 17, "Pancreatic Cancer": 7, "Prostate Cancer": 20, "Male Breast Cancer": 7, "Colorectal Cancer": 2},
    "PALB2":  {"Breast Cancer": 53, "Ovarian Cancer": 5,  "Pancreatic Cancer": 3},
    "BARD1":  {"Breast Cancer": 25, "Ovarian Cancer": 6},
    "BRIP1":  {"Breast Cancer": 20, "Ovarian Cancer": 13},
    "RAD51C": {"Breast Cancer": 18, "Ovarian Cancer": 11},
    "RAD51D": {"Breast Cancer": 15, "Ovarian Cancer": 13},
    "RAD51B": {"Breast Cancer": 20, "Ovarian Cancer": 5},
    "CHEK2":  {"Breast Cancer": 28, "Colorectal Cancer": 5, "Prostate Cancer": 15, "Ovarian Cancer": 4},
    "ATM":    {"Breast Cancer": 33, "Pancreatic Cancer": 6, "Prostate Cancer": 20, "Colorectal Cancer": 2},
    "NBN":    {"Breast Cancer": 20, "Ovarian Cancer": 5,  "Prostate Cancer": 12},
    "MRE11":  {"Breast Cancer": 18, "Ovarian Cancer": 5},
    "RAD50":  {"Breast Cancer": 20, "Ovarian Cancer": 5},
    "ABRAXAS1":{"Breast Cancer": 25, "Ovarian Cancer": 8},
    "XRCC2":  {"Breast Cancer": 22, "Ovarian Cancer": 5},
    "MBD4":   {"Breast Cancer": 15, "Colorectal Cancer": 10},

    # Lynch / MMR / Polymerase
    "MLH1":   {"Colorectal Cancer": 52, "Endometrial Cancer": 41, "Ovarian Cancer": 10, "Gastric Cancer": 13, "Urinary Tract Cancer": 8, "Pancreatic Cancer": 4},
    "MSH2":   {"Colorectal Cancer": 57, "Endometrial Cancer": 47, "Ovarian Cancer": 17, "Gastric Cancer": 15, "Urinary Tract Cancer": 28, "Pancreatic Cancer": 5},
    "MSH6":   {"Colorectal Cancer": 20, "Endometrial Cancer": 44, "Ovarian Cancer": 10, "Urinary Tract Cancer": 12},
    "PMS2":   {"Colorectal Cancer": 15, "Endometrial Cancer": 15, "Ovarian Cancer": 5},
    "PMS1":   {"Colorectal Cancer": 15, "Endometrial Cancer": 10},
    "MLH3":   {"Colorectal Cancer": 10, "Endometrial Cancer": 10},
    "MSH3":   {"Colorectal Cancer": 10, "Endometrial Cancer": 8},
    "POLE":   {"Colorectal Cancer": 70, "Endometrial Cancer": 60, "Ovarian Cancer": 15},
    "POLD1":  {"Colorectal Cancer": 30, "Endometrial Cancer": 25},
    "EPCAM":  {"Colorectal Cancer": 75, "Endometrial Cancer": 25, "Ovarian Cancer": 12},

    # Polyposis / Hamartoma
    "APC":    {"Colorectal Cancer": 93, "Duodenal Cancer": 10, "Gastric Cancer": 5, "Thyroid Cancer": 2},
    "MUTYH":  {"Colorectal Cancer": 65, "Duodenal Cancer": 4,  "Ovarian Cancer": 3, "Bladder Cancer": 3},
    "NTHL1":  {"Colorectal Cancer": 45, "Breast Cancer": 20, "Endometrial Cancer": 10},
    "BMPR1A": {"Colorectal Cancer": 40, "Gastric Cancer": 20, "Pancreatic Cancer": 5},
    "SMAD4":  {"Colorectal Cancer": 40, "Gastric Cancer": 20, "Pancreatic Cancer": 10},
    "GREM1":  {"Colorectal Cancer": 40, "Ovarian Cancer": 5},
    "RNF43":  {"Colorectal Cancer": 30, "Pancreatic Cancer": 15},
    "AXIN2":  {"Colorectal Cancer": 30, "Breast Cancer": 20},
    "PTEN":   {"Breast Cancer": 85, "Thyroid Cancer": 35, "Endometrial Cancer": 28, "Kidney Cancer": 34, "Colorectal Cancer": 9},
    "STK11":  {"Colorectal Cancer": 39, "Gastric Cancer": 29, "Breast Cancer": 54, "Pancreatic Cancer": 36, "Lung Cancer": 15, "Ovarian Cancer": 21},

    # Li-Fraumeni
    "TP53":   {"Breast Cancer": 54, "Soft Tissue Sarcoma": 22, "Brain Tumor": 14, "Adrenal Cortex Cancer": 9, "Leukemia": 7, "Osteosarcoma": 7, "Colorectal Cancer": 5, "Lung Cancer": 4},

    # Renal / VHL
    "VHL":    {"Kidney Cancer (ccRCC)": 70, "Hemangioblastoma (CNS)": 60, "Pheochromocytoma": 20, "Pancreatic Neuroendocrine": 15},
    "MET":    {"Kidney Cancer (papillary)": 70, "Gastric Cancer": 15},
    "FLCN":   {"Kidney Cancer": 27, "Lung (pneumothorax)": 80},
    "BAP1":   {"Mesothelioma": 25, "Uveal Melanoma": 25, "Kidney Cancer": 15, "Cutaneous Melanoma": 11, "Breast Cancer": 10},
    "FH":     {"Kidney Cancer (HLRCC)": 15, "Leiomyomatosis": 76},
    "SDHA":   {"Pheochromocytoma/Paraganglioma": 10, "Gastric Cancer": 15},
    "SDHB":   {"Pheochromocytoma/Paraganglioma": 29, "Kidney Cancer": 14, "Gastric Cancer": 7, "Thyroid Cancer": 5},
    "SDHC":   {"Pheochromocytoma/Paraganglioma": 20, "Kidney Cancer": 5},
    "SDHD":   {"Pheochromocytoma/Paraganglioma": 43, "Kidney Cancer": 5, "Thyroid Cancer": 5},
    "SDHAF2": {"Pheochromocytoma/Paraganglioma": 30},

    # Melanoma / Telomere
    "CDKN2A": {"Melanoma": 76, "Pancreatic Cancer": 17},
    "CDK4":   {"Melanoma": 70, "Pancreatic Cancer": 10},
    "MITF":   {"Melanoma": 14, "Kidney Cancer": 6},
    "POT1":   {"Melanoma": 25, "Colorectal Cancer": 10, "Breast Cancer": 5},
    "ACD":    {"Melanoma": 20, "Breast Cancer": 5},
    "TERF2IP":{"Melanoma": 15},
    "TINF2":  {"Aplastic Anemia": 60, "Leukemia": 20, "Lung Cancer": 10},
    "TERT":   {"Aplastic Anemia": 40, "AML/MDS": 25, "Lung Cancer": 15, "Liver Cancer": 10, "Colorectal Cancer": 5},
    "PARN":   {"Aplastic Anemia": 30, "AML/MDS": 15},
    "RTEL1":  {"Aplastic Anemia": 40, "Lung Cancer": 10},

    # Endocrine
    "MEN1":   {"Parathyroid Tumor": 95, "Pancreatic Neuroendocrine": 54, "Pituitary Tumor": 42, "Adrenal Tumor": 40},
    "RET":    {"Thyroid Cancer (MTC)": 95, "Pheochromocytoma": 50, "Parathyroid Tumor": 25},

    # Gastric
    "CDH1":   {"Gastric Cancer (diffuse)": 70, "Breast Cancer (lobular)": 42, "Colorectal Cancer": 10},
    "CTNNA1": {"Gastric Cancer (diffuse)": 60},

    # Hematologic
    "RUNX1":  {"AML/MDS": 44, "Lymphoma": 15, "Breast Cancer": 5},
    "CEBPA":  {"AML": 65, "MDS": 20},
    "ETV6":   {"AML/MDS": 30, "Lymphoma": 20},
    "ANKRD26":{"AML/MDS": 20, "Thrombocytopenia": 100},
    "DDX41":  {"AML/MDS": 50, "Lymphoma": 20},
    "GATA2":  {"AML/MDS": 76, "Lymphedema": 50},
    "SAMD9":  {"AML/MDS": 30},
    "SAMD9L": {"AML/MDS": 30},

    # Default fallback for any other hereditary gene
    "GEN_DEFAULT": {"Gene-Associated Cancer": 35},
}

POPULATION_BASELINE = {
    "Breast Cancer":                  "12-13%",
    "Ovarian Cancer":                 "1-2%",
    "Colorectal Cancer":              "4-5%",
    "Pancreatic Cancer":              "1-2%",
    "Prostate Cancer":                "11-13%",
    "Melanoma":                       "2-3%",
    "Thyroid Cancer":                 "1-2%",
    "Kidney Cancer":                  "1-2%",
    "Gastric Cancer":                 "1-2%",
    "Endometrial Cancer":             "2-3%",
    "Lung Cancer":                    "5-6%",
    "Soft Tissue Sarcoma":            "<1%",
    "Brain Tumor":                    "<1%",
    "AML/MDS":                        "<1%",
    "Leukemia":                       "<1%",
    "Pheochromocytoma/Paraganglioma": "<0.1%",
    "Osteosarcoma":                   "<1%",
    "Mesothelioma":                   "<0.5%",
    "Retinoblastoma":                 "<0.01%",
    "Urinary Tract Cancer":           "1-2%",
}

# ─────────────────────────────────────────────────────────────
# 2. RDKIT & 190-PAIR GRANTHAM CONSTANTS (Matching build_v2_features.py)
# ─────────────────────────────────────────────────────────────
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'TER': '*', '*': '*'
}

AA_SMILES = {
    'A': 'CC(N)C(=O)O', 'R': 'NC(CCCNC(=N)N)C(=O)O', 'N': 'NC(CC(=O)N)C(=O)O',
    'D': 'NC(CC(=O)O)C(=O)O', 'C': 'NC(CS)C(=O)O', 'Q': 'NC(CCC(=O)N)C(=O)O',
    'E': 'NC(CCC(=O)O)C(=O)O', 'G': 'NCC(=O)O', 'H': 'NC(Cc1c[nH]cn1)C(=O)O',
    'I': 'CCC(C)C(N)C(=O)O', 'L': 'CC(C)CC(N)C(=O)O', 'K': 'NCCCCC(N)C(=O)O',
    'M': 'CSCCC(N)C(=O)O', 'F': 'NC(Cc1ccccc1)C(=O)O', 'P': 'O=C(O)C1CCCN1',
    'S': 'NC(CO)C(=O)O', 'T': 'CC(O)C(N)C(=O)O', 'W': 'NC(Cc1c[nH]c2ccccc12)C(=O)O',
    'Y': 'NC(Cc1ccc(O)cc1)C(=O)O', 'V': 'CC(C)C(N)C(=O)O',
}

AA_CHARGE = {
    'R': 1.0, 'K': 1.0, 'H': 0.1, 'D': -1.0, 'E': -1.0,
    'A': 0.0, 'N': 0.0, 'C': 0.0, 'Q': 0.0, 'G': 0.0,
    'I': 0.0, 'L': 0.0, 'M': 0.0, 'F': 0.0, 'P': 0.0,
    'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0,
}

GRANTHAM_CONSTANTS = {
    'S': (1.42, 9.2, 32.0), 'R': (0.65, 10.5, 124.0), 'L': (0.0, 4.9, 111.0),
    'P': (0.39, 8.0, 32.5), 'T': (0.71, 8.6, 61.0),  'A': (0.0, 8.1, 31.0),
    'V': (0.0, 5.9, 84.0),  'G': (0.74, 9.0, 3.0),   'I': (0.0, 5.2, 111.0),
    'F': (0.0, 5.2, 132.0), 'Y': (0.20, 6.2, 136.0), 'C': (2.75, 5.5, 55.0),
    'H': (0.58, 10.4, 96.0),'Q': (0.89, 10.5, 85.0), 'N': (1.33, 11.6, 56.0),
    'K': (0.33, 11.3, 119.0),'D': (1.38, 13.0, 54.0),'E': (0.92, 12.3, 83.0),
    'M': (0.0, 5.7, 105.0), 'W': (0.13, 5.4, 170.0),
}

def build_grantham_matrix():
    matrix = {}
    for a1, (c1, p1, v1) in GRANTHAM_CONSTANTS.items():
        for a2, (c2, p2, v2) in GRANTHAM_CONSTANTS.items():
            if a1 == a2:
                matrix[(a1, a2)] = 0.0
            else:
                d = 50.723 * math.sqrt(
                    1.833 * (c1 - c2)**2 + 0.1018 * (p1 - p2)**2 + 0.000399 * (v1 - v2)**2
                )
                matrix[(a1, a2)] = round(d, 1)
    return matrix

GRANTHAM_MATRIX = build_grantham_matrix()

def compute_rdkit_aa_properties():
    props = {}
    for code, smi in AA_SMILES.items():
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        props[code] = {
            'mw': float(Descriptors.ExactMolWt(mol)),
            'logp': float(Descriptors.MolLogP(mol)),
            'tpsa': float(Descriptors.TPSA(mol)),
            'charge': AA_CHARGE[code],
            'fp': fp
        }
    return props

AA_PROPS = compute_rdkit_aa_properties()
TRANSITIONS = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}

# ─────────────────────────────────────────────────────────────
# 3. MODEL ARCHITECTURE (Must match train_resnet_v2.py)
# ─────────────────────────────────────────────────────────────
class ResBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.Dropout(dropout),
        )
    def forward(self, x):
        return torch.nn.functional.gelu(x + self.block(x))

class GenomeDLResNetV2(nn.Module):
    def __init__(self, n_cont=18, n_genes=130, n_syn=15, n_vclass=11,
                 emb_gene=16, emb_syn=8, emb_vcls=8, hidden=256, n_blocks=2,
                 dropout=0.2, head_drop=0.1):
        super().__init__()
        self.gene_emb  = nn.Embedding(n_genes,  emb_gene)
        self.syn_emb   = nn.Embedding(n_syn,    emb_syn)
        self.vcls_emb  = nn.Embedding(n_vclass, emb_vcls)
        in_dim = n_cont + emb_gene + emb_syn + emb_vcls
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(dropout))
        self.res_blocks = nn.Sequential(*[ResBlock(hidden, dropout) for _ in range(n_blocks)])
        self.head = nn.Sequential(
            nn.Linear(hidden, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(head_drop),
            nn.Linear(64, 1))
    def forward(self, x_cont, x_gene, x_syn, x_vcls):
        e = torch.cat([x_cont, self.gene_emb(x_gene),
                       self.syn_emb(x_syn), self.vcls_emb(x_vcls)], dim=1)
        return self.head(self.res_blocks(self.input_proj(e))).squeeze(1)

# ─────────────────────────────────────────────────────────────
# 4. FEATURE EXTRACTION PIPELINE
# ─────────────────────────────────────────────────────────────
GENE_SYNDROME_MAP = {
    **{g: "Breast_Ovarian_HR_repair" for g in [
        "BRCA1","BRCA2","PALB2","BARD1","BRIP1","RAD51C","RAD51D",
        "RAD51B","CHEK2","ATM","NBN","MRE11","RAD50","ABRAXAS1","XRCC2","MBD4"]},
    **{g: "Lynch_MMR_polymerase" for g in [
        "MLH1","MSH2","MSH6","PMS2","PMS1","MLH3","MSH3","POLE","POLD1","EPCAM"]},
    "TP53": "Li_Fraumeni",
    **{g: "Polyposis_hamartoma" for g in [
        "APC","MUTYH","NTHL1","BMPR1A","SMAD4","GREM1","RNF43","AXIN2","PTEN","STK11"]},
    **{g: "Renal" for g in ["VHL","MET","FLCN","BAP1","FH",
                             "SDHA","SDHB","SDHC","SDHD","SDHAF2","TMEM127","MAX"]},
    **{g: "Melanoma_skin_telomere" for g in [
        "CDKN2A","CDK4","MITF","POT1","ACD","TERF2IP","TINF2","TERT","PARN","RTEL1"]},
    **{g: "Endocrine_MEN" for g in ["MEN1","RET","PRKAR1A","CDKN1B","AIP"]},
    **{g: "Hereditary_gastric" for g in ["CDH1","CTNNA1"]},
    **{g: "Hematologic_predisposition" for g in [
        "RUNX1","CEBPA","ETV6","ANKRD26","DDX41","GATA2","SAMD9","SAMD9L","CBL","PTPN11"]},
    **{g: "Fanconi_DNA_repair_chromosome_instability" for g in [
        "FANCA","FANCB","FANCC","FANCD2","FANCE","FANCF","FANCG","FANCI","FANCL",
        "FANCM","SLX4","ERCC2","ERCC3","ERCC4","ERCC5","XPA","XPC","DDB2","POLH",
        "BLM","WRN","RECQL","RECQL4","BUB1B","FAN1","NBN"]},
    **{g: "Neurocutaneous_rhabdoid_DICER1" for g in [
        "NF1","NF2","SMARCB1","SMARCE1","SMARCA4","SUFU","PTCH1","PTCH2",
        "DICER1","LZTR1","SPRED1"]},
    **{g: "Paraganglioma_pheo" for g in ["SDHB","SDHC","SDHD","SDHA","SDHAF2","TMEM127","MAX","VHL"]},
    **{g: "Xeroderma_NER" for g in ["XPA","XPC","DDB2","ERCC2","ERCC3","ERCC4","ERCC5","POLH"]},
    **{g: "Retinoblastoma_Wilms_overgrowth" for g in [
        "RB1","WT1","DIS3L2","TRIP13","REST","CEP57","CDKN1C","SBDS","DKC1"]},
}

# Canonical approx gene lengths
APPROX_LENGTHS = {
    "BRCA1": 1864, "BRCA2": 3419, "TP53": 394, "PTEN": 404, "MLH1": 757,
    "MSH2": 935, "MSH6": 1361, "APC": 2844, "ATM": 3057, "CDH1": 883
}

def normalize_aa(aa_str):
    if not aa_str:
        return None
    s = str(aa_str).strip().upper()
    if len(s) == 3 and s in THREE_TO_ONE:
        return THREE_TO_ONE[s]
    if len(s) == 1 and (s in AA_PROPS or s == '*'):
        return s
    return None

def build_features(gene, ref_aa, alt_aa, pos, ref_base, alt_base,
                   variant_class, ckpt, on_primary=1):
    gene_to_id     = ckpt["gene_to_id"]
    syndrome_to_id = ckpt["syndrome_to_id"]
    class_to_id    = ckpt["class_to_id"]
    scaler_mean    = np.array(ckpt["scaler_mean"])
    scaler_scale   = np.array(ckpt["scaler_scale"])

    gene_id = gene_to_id.get(gene, 0)
    syn_name = GENE_SYNDROME_MAP.get(gene, "Other_hereditary")
    syn_id = syndrome_to_id.get(syn_name, 0)
    vcls_id = class_to_id.get(variant_class, 1)

    ra = normalize_aa(ref_aa)
    aa = normalize_aa(alt_aa)

    is_standard_ra = ra in AA_PROPS
    is_standard_aa = aa in AA_PROPS

    g_len = APPROX_LENGTHS.get(gene, 1000)

    if is_standard_ra and is_standard_aa:
        has_protein = 1
        rel_pos = min(1.0, max(0.0, pos / g_len)) if pos > 0 else 0.0
        g_score = GRANTHAM_MATRIX.get((ra, aa), 100.0)
        p_ref = AA_PROPS[ra]
        p_alt = AA_PROPS[aa]
        d_mw = round(p_alt['mw'] - p_ref['mw'], 3)
        d_charge = round(p_alt['charge'] - p_ref['charge'], 3)
        d_logp = round(p_alt['logp'] - p_ref['logp'], 3)
        d_tpsa = round(p_alt['tpsa'] - p_ref['tpsa'], 3)
        if ra == aa:
            tanimoto = 0.0
        else:
            sim = DataStructs.TanimotoSimilarity(p_ref['fp'], p_alt['fp'])
            tanimoto = round(1.0 - sim, 4)
    elif aa == '*' or ra == '*' or variant_class in ('nonsense', 'start_lost'):
        has_protein = 1
        rel_pos = min(1.0, max(0.0, pos / g_len)) if pos > 0 else 0.0
        g_score = 200.0
        if is_standard_ra:
            p_ref = AA_PROPS[ra]
            d_mw = -p_ref['mw']
            d_charge = -p_ref['charge']
            d_logp = -p_ref['logp']
            d_tpsa = -p_ref['tpsa']
        else:
            d_mw = d_charge = d_logp = d_tpsa = 0.0
        tanimoto = 1.0
    else:
        has_protein = 0
        rel_pos = 0.0
        g_score = d_mw = d_charge = d_logp = d_tpsa = tanimoto = 0.0

    r_base = ref_base.upper()
    a_base = alt_base.upper()
    is_trans = 1 if (r_base, a_base) in TRANSITIONS else 0

    bases = ['A', 'C', 'G', 'T']
    ref_oh = [1 if r_base == b else 0 for b in bases]
    alt_oh = [1 if a_base == b else 0 for b in bases]

    scale_vals = [g_score, d_mw, d_charge, d_logp, d_tpsa, tanimoto, rel_pos]
    scale_arr = (np.array(scale_vals) - scaler_mean) / scaler_scale
    binary_vals = [has_protein, is_trans, on_primary] + ref_oh + alt_oh

    cont = np.concatenate([scale_arr, binary_vals]).astype(np.float32)

    x_cont = torch.tensor(cont, dtype=torch.float32).unsqueeze(0)
    x_gene = torch.tensor([gene_id], dtype=torch.long)
    x_syn  = torch.tensor([syn_id],  dtype=torch.long)
    x_vcls = torch.tensor([vcls_id], dtype=torch.long)

    return x_cont, x_gene, x_syn, x_vcls, g_score

# ─────────────────────────────────────────────────────────────
# 5. USER REPORT
# ─────────────────────────────────────────────────────────────
def bar(pct, width=24):
    filled = int(round(min(100.0, max(0.0, pct)) / 100 * width))
    return "#" * filled + "-" * (width - filled)

def print_report(gene, ref_aa, alt_aa, pos, ref_base, alt_base,
                 variant_class, p_path, grantham_score):
    p_pct = p_path * 100

    gene_pen = PENETRANCE.get(gene, PENETRANCE["GEN_DEFAULT"])
    cancer_risks = {cancer: (p_path * pen) for cancer, pen in gene_pen.items()}
    cancer_risks_sorted = sorted(cancer_risks.items(), key=lambda x: -x[1])

    if p_pct >= 85:
        risk_label = "HIGH RISK - LIKELY PATHOGENIC"
        badge = "[!!!]"
    elif p_pct >= 50:
        risk_label = "MODERATE-HIGH RISK - PATHOGENIC TENDENCY"
        badge = "[!!]"
    elif p_pct >= 20:
        risk_label = "UNCERTAIN SIGNIFICANCE (VUS range)"
        badge = "[!]"
    else:
        risk_label = "LOW RISK - LIKELY BENIGN"
        badge = "[OK]"

    ra = normalize_aa(ref_aa) or ref_aa or ""
    aa = normalize_aa(alt_aa) or alt_aa or ""
    aa_str = f"p.{ra}{pos}{aa}" if ra and aa and pos else "N/A"
    syndrome = GENE_SYNDROME_MAP.get(gene, "Hereditary Cancer").replace("_", " ")

    W = 66
    print("=" * W)
    print(f"  GenomeDL v2 -- Cancer Risk & Pathogenicity Report")
    print("=" * W)
    print(f"  Gene              : {gene}  ({syndrome})")
    print(f"  Variant           : {aa_str}  ({ref_base}>{alt_base})")
    print(f"  Variant Class     : {variant_class}")
    if grantham_score > 0:
        sev = "radical" if grantham_score >= 100 else "moderate" if grantham_score >= 50 else "conservative"
        print(f"  Grantham Score    : {grantham_score:.1f}  ({sev} amino acid exchange)")
    print("-" * W)
    print(f"  PATHOGENICITY PROBABILITY: {p_pct:5.1f}%  {badge} {risk_label}")
    print(f"  (Model likelihood of variant being disease-causing)")
    print("=" * W)
    print(f"  LIFETIME CANCER RISK ESTIMATE PER ORGAN")
    print(f"  Formula: Cancer Risk (%) = P(pathogenic) x Gene Penetrance (%)")
    print("-" * W)
    print(f"  {'Cancer Type':<30} {'Est. Risk':>10}   {'Population Baseline'}")
    print("-" * W)

    for cancer, risk in cancer_risks_sorted:
        base = POPULATION_BASELINE.get(cancer, "<varies>")
        b = bar(risk)
        print(f"  {cancer:<30} {risk:8.1f}%   (Pop: {base})")
        print(f"  {'':<30} [{b}]")

    print("=" * W)
    print(f"  CLINICAL CONTEXT & DISCLAIMER")
    print(f"  - Penetrance numbers reflect lifelong incidence in mutation carriers.")
    print(f"  - If variant is benign, organ risk remains near population baseline.")
    print(f"  - Research prototype for in-silico screening. NOT a medical diagnosis.")
    print("=" * W)

def load_model(weights_path):
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    model = GenomeDLResNetV2(
        n_cont=len(ckpt["cont_cols"]),
        n_genes=ckpt["n_genes"],
        n_syn=ckpt["n_syn"],
        n_vclass=ckpt["n_vclass"],
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt

def predict_one(model, ckpt, gene, ref_aa, alt_aa, pos, ref_base, alt_base,
                variant_class="missense", on_primary=1):
    x_cont, x_gene, x_syn, x_vcls, g_score = build_features(
        gene, ref_aa, alt_aa, pos, ref_base, alt_base,
        variant_class, ckpt, on_primary)
    with torch.no_grad():
        logit = model(x_cont, x_gene, x_syn, x_vcls)
    p = torch.sigmoid(logit).item()
    return p, g_score

def demo_mode(model, ckpt):
    print("\n" + "="*66)
    print("  RUNNING COMPREHENSIVE GENOMEDL DEMO ACROSS 4 MAJOR GENES")
    print("="*66 + "\n")

    cases = [
        ("BRCA1", "Cys", "Gly", 61,  "A", "C", "missense", "High-penetrance BRCA1 RING domain missense (Breast/Ovary)"),
        ("MLH1",  None,  None,  0,   "C", "T", "nonsense", "MLH1 Truncating stop codon (Lynch Syndrome / Colon)"),
        ("BRCA2", "Val", "Ile", 600, "G", "A", "missense", "Conservative BRCA2 missense (Likely Benign)"),
        ("TP53",  "Arg", "His", 175, "C", "T", "missense", "Classic TP53 DNA-binding hotspot (Li-Fraumeni)"),
    ]

    for gene, ref_aa, alt_aa, pos, ref, alt, vcls, desc in cases:
        print(f"\n[Case Study]: {desc}")
        p, g = predict_one(model, ckpt, gene, ref_aa, alt_aa, pos, ref, alt, vcls)
        print_report(gene, ref_aa, alt_aa, pos, ref, alt, vcls, p, g)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenomeDL v2 — Cancer Risk & Pathogenicity Predictor")
    parser.add_argument("--weights",       default="models/genome_dl_resnet_v2.pth")
    parser.add_argument("--gene",          default="BRCA1")
    parser.add_argument("--ref_aa",        default=None, help="e.g. Cys or C")
    parser.add_argument("--alt_aa",        default=None, help="e.g. Gly or G")
    parser.add_argument("--pos",           type=int, default=0, help="amino acid codon position")
    parser.add_argument("--ref",           default="A", help="reference nucleotide (plus-strand)")
    parser.add_argument("--alt",           default="C", help="alternate nucleotide (plus-strand)")
    parser.add_argument("--variant_class", default="missense",
                        choices=["missense","nonsense","synonymous","splice_canonical",
                                 "splice_region","intronic","utr5","utr3","start_lost",
                                 "other_noncoding","coding_no_protein_annot"])
    parser.add_argument("--demo",          action="store_true", help="run 4 built-in demo case studies")
    args = parser.parse_args()

    weights = Path(args.weights)
    if not weights.exists():
        print(f"ERROR: Checkpoint not found at {weights}")
        sys.exit(1)

    model, ckpt = load_model(str(weights))

    if args.demo:
        demo_mode(model, ckpt)
    else:
        p, g = predict_one(model, ckpt,
                           gene=args.gene,
                           ref_aa=args.ref_aa,
                           alt_aa=args.alt_aa,
                           pos=args.pos,
                           ref_base=args.ref,
                           alt_base=args.alt,
                           variant_class=args.variant_class)
        print_report(args.gene, args.ref_aa, args.alt_aa, args.pos,
                     args.ref, args.alt, args.variant_class, p, g)
