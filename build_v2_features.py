# ==============================================================================
# GenomeDL v2: High-Fidelity Feature Engineering Pipeline
# Computes full 190-pair Grantham matrix, RDKit chemical deltas, canonical 
# protein positions, and explicit molecular variant classes for 130 genes.
# ==============================================================================

import os
import sys
import math
import time
import argparse
import numpy as np
import pandas as pd

# Suppress RDKit deprecation warnings
import warnings
warnings.filterwarnings('ignore')
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors, AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

# ------------------------------------------------------------------------------
# 1. 20 Standard Amino Acids: RDKit Structures & Formal Charge (pH 7.4)
# ------------------------------------------------------------------------------
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

AA_CHARGE = {
    'R': 1.0, 'K': 1.0, 'H': 0.1,
    'D': -1.0, 'E': -1.0,
    'A': 0.0, 'N': 0.0, 'C': 0.0, 'Q': 0.0, 'G': 0.0,
    'I': 0.0, 'L': 0.0, 'M': 0.0, 'F': 0.0, 'P': 0.0,
    'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0,
}

# ------------------------------------------------------------------------------
# 2. Complete Grantham (1974) Constants: Full 190-Pair Matrix Formulation
# D = 50.723 * sqrt( 1.833*(c - c')^2 + 0.1018*(p - p')^2 + 0.000399*(v - v')^2 )
# ------------------------------------------------------------------------------
GRANTHAM_CONSTANTS = {
    'S': (1.42, 9.2, 32.0),
    'R': (0.65, 10.5, 124.0),
    'L': (0.0, 4.9, 111.0),
    'P': (0.39, 8.0, 32.5),
    'T': (0.71, 8.6, 61.0),
    'A': (0.0, 8.1, 31.0),
    'V': (0.0, 5.9, 84.0),
    'G': (0.74, 9.0, 3.0),
    'I': (0.0, 5.2, 111.0),
    'F': (0.0, 5.2, 132.0),
    'Y': (0.20, 6.2, 136.0),
    'C': (2.75, 5.5, 55.0),
    'H': (0.58, 10.4, 96.0),
    'Q': (0.89, 10.5, 85.0),
    'N': (1.33, 11.6, 56.0),
    'K': (0.33, 11.3, 119.0),
    'D': (1.38, 13.0, 54.0),
    'E': (0.92, 12.3, 83.0),
    'M': (0.0, 5.7, 105.0),
    'W': (0.13, 5.4, 170.0),
}

def build_grantham_matrix():
    matrix = {}
    for a1, (c1, p1, v1) in GRANTHAM_CONSTANTS.items():
        for a2, (c2, p2, v2) in GRANTHAM_CONSTANTS.items():
            if a1 == a2:
                matrix[(a1, a2)] = 0.0
            else:
                d = 50.723 * math.sqrt(
                    1.833 * (c1 - c2)**2 +
                    0.1018 * (p1 - p2)**2 +
                    0.000399 * (v1 - v2)**2
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

VARIANT_CLASSES = [
    'synonymous', 'missense', 'nonsense', 'start_lost',
    'splice_canonical', 'splice_region', 'intronic',
    'utr5', 'utr3', 'other_noncoding', 'coding_no_protein_annot'
]
CLASS_TO_ID = {c: i for i, c in enumerate(VARIANT_CLASSES)}

# ------------------------------------------------------------------------------
# 3. Main Feature Engineering Engine
# ------------------------------------------------------------------------------
def featurize_dataset(input_csv, output_csv):
    print("\n" + "="*80)
    print(f"[*] GenomeDL v2: Feature Engineering Pipeline")
    print(f"[*] Input Dataset:  {input_csv}")
    print(f"[*] Output Dataset: {output_csv}")
    print("="*80)
    
    t0 = time.time()
    df = pd.read_csv(input_csv)
    n_rows = len(df)
    print(f"[+] Loaded {n_rows:,} variants across {df['gene'].nunique()} genes.")
    
    # 1. Map Canonical Protein Lengths per gene
    prim = df[df['on_primary_transcript'] == 1]
    gene_max_pos = prim.groupby('gene')['aa_pos'].max().to_dict()
    # Global fallback if not in primary
    all_max_pos = df.groupby('gene')['aa_pos'].max().to_dict()
    
    gene_lengths = {}
    for g in df['gene'].unique():
        length = gene_max_pos.get(g, 0)
        if length == 0:
            length = all_max_pos.get(g, 1000)
        gene_lengths[g] = max(length, 100) # At least 100 aa
        
    print(f"[+] Computed canonical protein lengths for {len(gene_lengths)} genes.")
    
    # 2. Categorical Encodings
    unique_genes = sorted(df['gene'].unique())
    gene_to_id = {g: i for i, g in enumerate(unique_genes)}
    
    unique_syndromes = sorted(df['syndrome_group'].unique())
    syn_to_id = {s: i for i, s in enumerate(unique_syndromes)}
    
    print(f"[+] Indexed {len(unique_genes)} Genes & {len(unique_syndromes)} Syndromes.")
    
    # Pre-allocate output feature arrays
    has_protein_change = np.zeros(n_rows, dtype=np.int32)
    rel_protein_pos = np.zeros(n_rows, dtype=np.float32)
    grantham_scores = np.zeros(n_rows, dtype=np.float32)
    delta_mw = np.zeros(n_rows, dtype=np.float32)
    delta_charge = np.zeros(n_rows, dtype=np.float32)
    delta_logp = np.zeros(n_rows, dtype=np.float32)
    delta_tpsa = np.zeros(n_rows, dtype=np.float32)
    tanimoto_dist = np.zeros(n_rows, dtype=np.float32)
    is_transition = np.zeros(n_rows, dtype=np.int32)
    
    gene_ids = np.zeros(n_rows, dtype=np.int32)
    syndrome_ids = np.zeros(n_rows, dtype=np.int32)
    variant_class_ids = np.zeros(n_rows, dtype=np.int32)
    
    ref_aa_col = df['ref_aa'].values
    alt_aa_col = df['alt_aa'].values
    aa_pos_col = df['aa_pos'].values
    gene_col = df['gene'].values
    syn_col = df['syndrome_group'].values
    vclass_col = df['variant_class'].values
    ref_col = df['ref'].values
    alt_col = df['alt'].values
    
    print(f"[*] Processing biophysical deltas and Grantham matrix across {n_rows:,} rows...")
    
    for i in range(n_rows):
        gene = gene_col[i]
        gene_ids[i] = gene_to_id[gene]
        syndrome_ids[i] = syn_to_id[syn_col[i]]
        variant_class_ids[i] = CLASS_TO_ID.get(vclass_col[i], CLASS_TO_ID['other_noncoding'])
        
        # DNA Transition check
        r_base, a_base = ref_col[i], alt_col[i]
        is_transition[i] = 1 if (r_base, a_base) in TRANSITIONS else 0
        
        # Protein change & Amino acids
        ra, aa = ref_aa_col[i], alt_aa_col[i]
        pos = aa_pos_col[i]
        
        # Check if valid standard amino acid replacement
        is_standard_ra = ra in AA_PROPS
        is_standard_aa = aa in AA_PROPS
        
        if is_standard_ra and is_standard_aa:
            has_protein_change[i] = 1
            g_len = gene_lengths[gene]
            rel_protein_pos[i] = min(1.0, max(0.0, pos / g_len))
            
            # Full Grantham matrix lookup
            grantham_scores[i] = GRANTHAM_MATRIX.get((ra, aa), 100.0)
            
            # RDKit Chemical deltas
            p_ref = AA_PROPS[ra]
            p_alt = AA_PROPS[aa]
            
            delta_mw[i] = round(p_alt['mw'] - p_ref['mw'], 3)
            delta_charge[i] = round(p_alt['charge'] - p_ref['charge'], 2)
            delta_logp[i] = round(p_alt['logp'] - p_ref['logp'], 3)
            delta_tpsa[i] = round(p_alt['tpsa'] - p_ref['tpsa'], 3)
            
            if ra == aa:
                tanimoto_dist[i] = 0.0
            else:
                sim = DataStructs.TanimotoSimilarity(p_ref['fp'], p_alt['fp'])
                tanimoto_dist[i] = round(1.0 - sim, 4)
                
        elif aa == '*' or ra == '*' or vclass_col[i] in ('nonsense', 'start_lost'):
            # Truncation / Nonsense / Start lost
            has_protein_change[i] = 1
            g_len = gene_lengths[gene]
            rel_protein_pos[i] = min(1.0, max(0.0, pos / g_len))
            grantham_scores[i] = 200.0 # Deterministic stop-loss severity
            
            if is_standard_ra:
                p_ref = AA_PROPS[ra]
                delta_mw[i] = -p_ref['mw']
                delta_charge[i] = -p_ref['charge']
                delta_logp[i] = -p_ref['logp']
                delta_tpsa[i] = -p_ref['tpsa']
            tanimoto_dist[i] = 1.0
        else:
            # Non-coding, intronic, UTR, splice
            has_protein_change[i] = 0
            rel_protein_pos[i] = 0.0
            grantham_scores[i] = 0.0
            delta_mw[i] = 0.0
            delta_charge[i] = 0.0
            delta_logp[i] = 0.0
            delta_tpsa[i] = 0.0
            tanimoto_dist[i] = 0.0

    # Add engineered features to dataframe
    df['has_protein_change'] = has_protein_change
    df['rel_protein_pos'] = rel_protein_pos
    df['grantham_score'] = grantham_scores
    df['delta_mw'] = delta_mw
    df['delta_charge'] = delta_charge
    df['delta_logp'] = delta_logp
    df['delta_tpsa'] = delta_tpsa
    df['tanimoto_chem_dist'] = tanimoto_dist
    df['is_transition'] = is_transition
    
    df['gene_id'] = gene_ids
    df['syndrome_id'] = syndrome_ids
    df['variant_class_id'] = variant_class_ids
    
    # One-hot nucleotide encoding
    for base in ['A', 'C', 'G', 'T']:
        df[f'ref_{base}'] = (df['ref'] == base).astype(np.int32)
        df[f'alt_{base}'] = (df['alt'] == base).astype(np.int32)
        
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False)
    
    elapsed = time.time() - t0
    file_size_mb = os.path.getsize(output_csv) / (1024 * 1024)
    print("\n" + "="*80)
    print(f"[+] Feature Engineering COMPLETE in {elapsed:.2f} seconds!")
    print(f"[+] Successfully generated: {output_csv} ({file_size_mb:.2f} MB)")
    print(f"[+] Total Rows: {len(df):,} | Total Columns: {len(df.columns)}")
    print("="*80)
    
    # Integrity check: verify Sarah's Cys->Gly Grantham score
    cys_gly_grantham = GRANTHAM_MATRIX.get(('C', 'G'), 0)
    print(f"\n[*] Validation Checks:")
    print(f"  • Cys->Gly Grantham Score: {cys_gly_grantham} (Expected: ~159, v1 bug was: 100)")
    print(f"  • Missense variants count: {df[df['variant_class'] == 'missense'].shape[0]:,}")
    print(f"  • Null values check: {df.isnull().sum().sum()} nulls found across dataset.")
    print("="*80 + "\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GenomeDL v2 Feature Engineering")
    parser.add_argument("--input", default="data/clinvar_labelled_snv.csv", help="Input clean ClinVar CSV")
    parser.add_argument("--output", default="data/v2_clinvar_featured_snv.csv", help="Output feature-engineered CSV")
    args = parser.parse_args()
    
    featurize_dataset(args.input, args.output)
