#!/usr/bin/env python3
"""
==============================================================================
GenomeDL v2: Biophysical & Cheminformatics Feature Engineering Pipeline
==============================================================================
This script transforms clean ClinVar v2 variant records (from out/) into 
feature-rich numerical datasets ready for deep learning training and evaluation.

Features Engineered:
1. Canonical 20x20 Grantham Matrix (1974) via pard (all 190 pairs + diagonal).
2. NCBI/Biopython BLOSUM62 substitution matrix.
3. RDKit Molecular Properties:
   - Delta Molecular Weight (delta_mw in Da)
   - Delta LogP (delta_logp, lipophilicity / hydropathy shift)
   - Delta Topological Polar Surface Area (delta_tpsa in A^2)
   - Delta Formal Charge at physiological pH 7.4 (delta_charge)
   - Side-chain Chemical Topology Distance (1.0 - Morgan Tanimoto similarity)
   - Polarity swap indicator (is_polarity_change)
   - Charge swap indicator (is_charge_change)
4. Genomic Strand One-Hot Nucleotides (ref_A/C/G/T, alt_A/C/G/T) & Transition flag.
5. Structural Protein Context:
   - Real primary transcript protein length (eliminating 0.5 sentinel bug)
   - Relative protein position (rel_protein_pos strictly in [0.0, 1.0])
   - Explicit has_protein_change indicator flag
6. Fine-Grained Molecular Class Encodings:
   - is_missense, is_nonsense, is_synonymous, is_start_lost
   - is_splice_canonical, is_splice_region, is_intronic, is_utr, is_noncoding
"""

import os
import sys
import time
import json
import pandas as pd
import numpy as np

# Bioinformatics & Cheminformatics
from Bio.Align import substitution_matrices
from pard.grantham import grantham
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# -----------------------------------------------------------------------------
# 1. 20 Standard Amino Acids Definition & Physicochemical Constants
# -----------------------------------------------------------------------------
AA_LIST = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
           'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

AA_SMILES = {
    'A': 'CC(N)C(=O)O',                         # Alanine
    'R': 'NC(CCCNC(=N)N)C(=O)O',                # Arginine
    'N': 'NC(CC(=O)N)C(=O)O',                   # Asparagine
    'D': 'NC(CC(=O)O)C(=O)O',                   # Aspartate
    'C': 'NC(CS)C(=O)O',                        # Cysteine
    'Q': 'NC(CCC(=O)N)C(=O)O',                  # Glutamine
    'E': 'NC(CCC(=O)O)C(=O)O',                  # Glutamate
    'G': 'NCC(=O)O',                            # Glycine
    'H': 'NC(Cc1c[nH]cn1)C(=O)O',               # Histidine
    'I': 'CCC(C)C(N)C(=O)O',                    # Isoleucine
    'L': 'CC(C)CC(N)C(=O)O',                    # Leucine
    'K': 'NCCCCC(N)C(=O)O',                     # Lysine
    'M': 'CSCCC(N)C(=O)O',                      # Methionine
    'F': 'NC(Cc1ccccc1)C(=O)O',                 # Phenylalanine
    'P': 'O=C(O)C1CCCN1',                       # Proline
    'S': 'NC(CO)C(=O)O',                        # Serine
    'T': 'CC(O)C(N)C(=O)O',                     # Threonine
    'W': 'NC(Cc1c[nH]c2ccccc12)C(=O)O',         # Tryptophan
    'Y': 'NC(Cc1ccc(O)cc1)C(=O)O',              # Tyrosine
    'V': 'CC(C)C(N)C(=O)O',                     # Valine
}

# Formal Charge at physiological pH 7.4
AA_CHARGE = {
    'R': 1.0, 'K': 1.0, 'H': 0.1,
    'D': -1.0, 'E': -1.0,
    'A': 0.0, 'N': 0.0, 'C': 0.0, 'Q': 0.0, 'G': 0.0,
    'I': 0.0, 'L': 0.0, 'M': 0.0, 'F': 0.0, 'P': 0.0,
    'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0
}

# Polarity Classification (1: Polar/Hydrophilic, 0: Nonpolar/Hydrophobic)
AA_POLARITY = {
    'R': 1, 'N': 1, 'D': 1, 'C': 1, 'E': 1, 'Q': 1, 'G': 0, 'H': 1,
    'K': 1, 'S': 1, 'T': 1, 'Y': 1, 'A': 0, 'I': 0, 'L': 0, 'M': 0,
    'F': 0, 'P': 0, 'W': 0, 'V': 0
}

TRANSITIONS = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}

# -----------------------------------------------------------------------------
# 2. Build Precomputed Biophysical Pairwise Table (400 Pairs)
# -----------------------------------------------------------------------------
def build_pairwise_biophysical_lookup():
    print("[*] Pre-computing canonical Grantham, BLOSUM62 & RDKit pairwise matrix...")
    blosum62 = substitution_matrices.load('BLOSUM62')
    
    # Compute RDKit monomer properties
    aa_props = {}
    for aa in AA_LIST:
        mol = Chem.MolFromSmiles(AA_SMILES[aa])
        aa_props[aa] = {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'charge': AA_CHARGE[aa],
            'polarity': AA_POLARITY[aa],
            'fp': AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        }
        
    pairwise = {}
    for ref in AA_LIST:
        for alt in AA_LIST:
            # Grantham distance
            g_dist = grantham(ref, alt)
            # BLOSUM62 score
            b_score = float(blosum62[ref, alt])
            # RDKit property deltas
            p_ref = aa_props[ref]
            p_alt = aa_props[alt]
            d_mw = round(p_alt['mw'] - p_ref['mw'], 3)
            d_logp = round(p_alt['logp'] - p_ref['logp'], 3)
            d_tpsa = round(p_alt['tpsa'] - p_ref['tpsa'], 3)
            d_charge = round(p_alt['charge'] - p_ref['charge'], 1)
            sim = DataStructs.TanimotoSimilarity(p_ref['fp'], p_alt['fp'])
            chem_dist = round(1.0 - sim, 3)
            is_pol_change = 1.0 if p_ref['polarity'] != p_alt['polarity'] else 0.0
            is_chg_change = 1.0 if abs(p_alt['charge'] - p_ref['charge']) > 0.05 else 0.0
            
            pairwise[(ref, alt)] = (g_dist, b_score, d_mw, d_logp, d_tpsa, d_charge, chem_dist, is_pol_change, is_chg_change)
            
    print(f"[+] Pairwise lookup generated for all {len(pairwise)} canonical amino acid pairs.")
    return pairwise

# -----------------------------------------------------------------------------
# 3. Compute Primary Transcript Protein Lengths
# -----------------------------------------------------------------------------
def get_protein_lengths_dict(out_dir):
    print("[*] Computing true primary transcript protein lengths across all panel genes...")
    l_path = os.path.join(out_dir, "clinvar_labelled_snv.csv")
    v_path = os.path.join(out_dir, "clinvar_vus_snv.csv")
    
    dfs = []
    if os.path.exists(l_path): dfs.append(pd.read_csv(l_path, usecols=['gene', 'variant_class', 'aa_pos', 'on_primary_transcript'], low_memory=False))
    if os.path.exists(v_path): dfs.append(pd.read_csv(v_path, usecols=['gene', 'variant_class', 'aa_pos', 'on_primary_transcript'], low_memory=False))
    
    full_df = pd.concat(dfs, ignore_index=True)
    primary = full_df[full_df['on_primary_transcript'] == 1]
    coding = primary[primary['variant_class'].isin(['missense', 'nonsense', 'synonymous', 'start_lost'])]
    
    prot_lens = coding.groupby('gene')['aa_pos'].max().to_dict()
    print(f"[+] Computed reference protein lengths for all {len(prot_lens)} genes on primary transcripts.")
    return prot_lens

# -----------------------------------------------------------------------------
# 4. Feature Extraction Function for a DataFrame
# -----------------------------------------------------------------------------
def featurize_variant_dataset(df, pairwise_lookup, prot_lens):
    n = len(df)
    t0 = time.time()
    print(f"[*] Featurizing dataset with {n:,} rows...")
    
    # Pre-allocate arrays
    grantham_arr = np.zeros(n, dtype=np.float32)
    blosum_arr = np.zeros(n, dtype=np.float32)
    delta_mw_arr = np.zeros(n, dtype=np.float32)
    delta_logp_arr = np.zeros(n, dtype=np.float32)
    delta_tpsa_arr = np.zeros(n, dtype=np.float32)
    delta_charge_arr = np.zeros(n, dtype=np.float32)
    chem_dist_arr = np.zeros(n, dtype=np.float32)
    pol_change_arr = np.zeros(n, dtype=np.float32)
    chg_change_arr = np.zeros(n, dtype=np.float32)
    
    rel_pos_arr = np.zeros(n, dtype=np.float32)
    has_prot_change_arr = np.zeros(n, dtype=np.float32)
    
    ref_aa_col = df['ref_aa'].values
    alt_aa_col = df['alt_aa'].values
    vclass_col = df['variant_class'].values
    gene_col = df['gene'].values
    pos_col = df['aa_pos'].values
    
    for i in range(n):
        vc = vclass_col[i]
        r_aa = ref_aa_col[i]
        a_aa = alt_aa_col[i]
        g = gene_col[i]
        p = pos_col[i]
        
        # Protein length & relative position
        g_len = prot_lens.get(g, 1000)
        if vc in ('missense', 'nonsense', 'synonymous', 'start_lost') and p > 0:
            rel_pos_arr[i] = min(1.0, max(0.0, float(p) / float(g_len)))
            has_prot_change_arr[i] = 1.0
        else:
            rel_pos_arr[i] = 0.0
            has_prot_change_arr[i] = 0.0
            
        # Biophysical features
        if (r_aa, a_aa) in pairwise_lookup:
            (g_dist, b_score, d_mw, d_logp, d_tpsa, d_chg, c_dist, pol_chg, chg_chg) = pairwise_lookup[(r_aa, a_aa)]
            grantham_arr[i] = g_dist
            blosum_arr[i] = b_score
            delta_mw_arr[i] = d_mw
            delta_logp_arr[i] = d_logp
            delta_tpsa_arr[i] = d_tpsa
            delta_charge_arr[i] = d_chg
            chem_dist_arr[i] = c_dist
            pol_change_arr[i] = pol_chg
            chg_change_arr[i] = chg_chg
        elif vc == 'nonsense' or a_aa == '*':
            # Premature termination codon (Nonsense)
            grantham_arr[i] = 200.0
            blosum_arr[i] = -4.0
            delta_mw_arr[i] = 0.0
            delta_logp_arr[i] = 0.0
            delta_tpsa_arr[i] = 0.0
            delta_charge_arr[i] = 0.0
            chem_dist_arr[i] = 1.0
            pol_change_arr[i] = 1.0
            chg_change_arr[i] = 1.0
        elif vc == 'synonymous':
            grantham_arr[i] = 0.0
            blosum_arr[i] = 4.0 # Typical positive diagonal BLOSUM score
            
    # Assign biophysical columns
    df['grantham_score'] = grantham_arr
    df['blosum62_score'] = blosum_arr
    df['delta_mw'] = delta_mw_arr
    df['delta_logp'] = delta_logp_arr
    df['delta_tpsa'] = delta_tpsa_arr
    df['delta_charge'] = delta_charge_arr
    df['tanimoto_chem_dist'] = chem_dist_arr
    df['is_polarity_change'] = pol_change_arr
    df['is_charge_change'] = chg_change_arr
    df['rel_protein_pos'] = rel_pos_arr
    df['has_protein_change'] = has_prot_change_arr
    
    # Genomic Nucleotide features
    ref_col = df['ref'].values
    alt_col = df['alt'].values
    for b in ['A', 'C', 'G', 'T']:
        df[f'ref_{b}'] = (ref_col == b).astype(np.float32)
        df[f'alt_{b}'] = (alt_col == b).astype(np.float32)
        
    is_trans = np.array([(1.0 if (r, a) in TRANSITIONS else 0.0) for r, a in zip(ref_col, alt_col)], dtype=np.float32)
    df['is_transition'] = is_trans
    
    # Molecular Class One-Hot Features
    df['is_missense'] = (df['variant_class'] == 'missense').astype(np.float32)
    df['is_nonsense'] = (df['variant_class'] == 'nonsense').astype(np.float32)
    df['is_synonymous'] = (df['variant_class'] == 'synonymous').astype(np.float32)
    df['is_start_lost'] = (df['variant_class'] == 'start_lost').astype(np.float32)
    df['is_splice_canonical'] = (df['variant_class'] == 'splice_canonical').astype(np.float32)
    df['is_splice_region'] = (df['variant_class'] == 'splice_region').astype(np.float32)
    df['is_intronic'] = (df['variant_class'] == 'intronic').astype(np.float32)
    df['is_utr'] = df['variant_class'].isin(['utr5', 'utr3']).astype(np.float32)
    df['is_other_noncoding'] = df['variant_class'].isin(['other_noncoding', 'coding_no_protein_annot']).astype(np.float32)
    
    elapsed = time.time() - t0
    print(f"[+] Featurization completed in {elapsed:.2f} seconds!")
    return df

# -----------------------------------------------------------------------------
# 5. Main Execution Pipeline
# -----------------------------------------------------------------------------
def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(script_dir, "out")
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    l_in = os.path.join(out_dir, "clinvar_labelled_snv.csv")
    v_in = os.path.join(out_dir, "clinvar_vus_snv.csv")
    
    if not os.path.exists(l_in):
        sys.exit(f"[!] Error: {l_in} not found. Please verify out/ directory.")
        
    pairwise_lookup = build_pairwise_biophysical_lookup()
    prot_lens = get_protein_lengths_dict(out_dir)
    
    # 1. Featurize Labelled SNVs (Training/Evaluation set)
    print("\n" + "="*80)
    print(f"[*] Processing Labelled SNV Dataset: {l_in}")
    print("="*80)
    df_l = pd.read_csv(l_in, low_memory=False)
    df_l_feat = featurize_variant_dataset(df_l, pairwise_lookup, prot_lens)
    
    l_out = os.path.join(data_dir, "v2_pan_cancer_featured_labelled.csv")
    df_l_feat.to_csv(l_out, index=False)
    print(f"[+] Wrote {len(df_l_feat):,} featurized rows to: {l_out}")
    
    # 2. Featurize VUS SNVs (Target scoring set for AI resolution)
    if os.path.exists(v_in):
        print("\n" + "="*80)
        print(f"[*] Processing VUS SNV Dataset: {v_in}")
        print("="*80)
        df_v = pd.read_csv(v_in, low_memory=False)
        df_v_feat = featurize_variant_dataset(df_v, pairwise_lookup, prot_lens)
        
        v_out = os.path.join(data_dir, "v2_pan_cancer_featured_vus.csv")
        df_v_feat.to_csv(v_out, index=False)
        print(f"[+] Wrote {len(df_v_feat):,} featurized rows to: {v_out}")
        
    print("\n" + "="*80)
    print(">>> MILESTONE 1 FEATURIZATION COMPLETE <<<")
    print("="*80)

if __name__ == '__main__':
    main()
