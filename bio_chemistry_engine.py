# ==============================================================================
# Bio-Chemistry, RDKit, Evolutionary Selection & Mutual Information Engine
# Master Pan-Cancer Risk Pipeline
# ==============================================================================

import os
import math
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, DataStructs

BASE_DIR = r"D:\DL"
REF_PANEL_FNA = os.path.join(BASE_DIR, "unified_128_gene_reference_panel.fna")

# 1. Universal Genetic Code
GENETIC_CODE = {
    'ATA':'I', 'ATC':'I', 'ATT':'I', 'ATG':'M',
    'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACT':'T',
    'AAC':'N', 'AAT':'N', 'AAA':'K', 'AAG':'K',
    'AGC':'S', 'AGT':'S', 'AGA':'R', 'AGG':'R',
    'CTA':'L', 'CTC':'L', 'CTG':'L', 'CTT':'L',
    'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCT':'P',
    'CAC':'H', 'CAT':'H', 'CAA':'Q', 'CAG':'Q',
    'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGT':'R',
    'GTA':'V', 'GTC':'V', 'GTG':'V', 'GTT':'V',
    'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCT':'A',
    'GAC':'D', 'GAT':'D', 'GAA':'E', 'GAG':'E',
    'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGT':'G',
    'TCA':'S', 'TCC':'S', 'TCG':'S', 'TCT':'S',
    'TTC':'F', 'TTT':'F', 'TTA':'L', 'TTG':'L',
    'TAC':'Y', 'TAT':'Y', 'TAA':'*', 'TAG':'*',
    'TGC':'C', 'TGT':'C', 'TGA':'*', 'TGG':'W',
}

COMPLEMENT = {'A': 'T', 'T': 'A', 'C': 'G', 'G': 'C', 'N': 'N'}

def rev_comp(seq):
    return "".join(COMPLEMENT.get(b, b) for b in reversed(seq.upper()))

# 2. RDKit Molecular Structures (Zwitterionic / Neutral standard amino acids)
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
    '*': 'O'                                    # Stop codon
}

# Formal Charge map at physiological pH 7.4
AA_CHARGE = {
    'R': 1.0, 'K': 1.0, 'H': 0.1,
    'D': -1.0, 'E': -1.0,
    'A': 0.0, 'N': 0.0, 'C': 0.0, 'Q': 0.0, 'G': 0.0,
    'I': 0.0, 'L': 0.0, 'M': 0.0, 'F': 0.0, 'P': 0.0,
    'S': 0.0, 'T': 0.0, 'W': 0.0, 'Y': 0.0, 'V': 0.0,
    '*': 0.0
}

# Grantham Matrix (1974)
GRANTHAM = {
    ('S', 'R'): 110, ('S', 'L'): 145, ('S', 'P'): 74, ('S', 'T'): 58, ('S', 'A'): 99,
    ('S', 'V'): 124, ('S', 'G'): 56,  ('S', 'I'): 142, ('S', 'K'): 121, ('S', 'M'): 135,
    ('S', 'T'): 58,  ('S', 'C'): 112, ('S', 'N'): 46,  ('S', 'Q'): 68,  ('S', 'Y'): 144,
    ('S', 'W'): 177, ('S', 'F'): 155, ('S', 'D'): 65,  ('S', 'E'): 80,  ('S', 'H'): 89,
    ('R', 'K'): 26,  ('R', 'H'): 29,  ('R', 'Q'): 43,  ('R', 'E'): 54,  ('R', 'D'): 96,
    ('R', 'N'): 86,  ('R', 'W'): 101, ('R', 'Y'): 77,  ('R', 'F'): 97,  ('R', 'M'): 91,
    ('R', 'I'): 97,  ('R', 'L'): 102, ('R', 'V'): 96,  ('R', 'P'): 103, ('R', 'T'): 71,
    ('R', 'A'): 112, ('R', 'G'): 125, ('R', 'C'): 180, ('E', 'D'): 45,  ('E', 'Q'): 29,
    ('E', 'K'): 56,  ('E', 'V'): 121, ('E', 'A'): 107, ('E', 'G'): 98,  ('E', 'H'): 40,
    ('L', 'I'): 5,   ('L', 'V'): 32,  ('L', 'M'): 15,  ('L', 'F'): 22,  ('V', 'I'): 29,
    ('V', 'M'): 21,  ('F', 'Y'): 22,  ('F', 'W'): 18,  ('Y', 'W'): 37,  ('D', 'N'): 23,
    ('Q', 'N'): 42,  ('Q', 'K'): 53,  ('P', 'A'): 27,  ('G', 'A'): 60,  ('C', 'A'): 195,
}

def get_grantham(a1, a2):
    if a1 == a2: return 0
    if a1 == '*' or a2 == '*': return 200
    if (a1, a2) in GRANTHAM: return GRANTHAM[(a1, a2)]
    if (a2, a1) in GRANTHAM: return GRANTHAM[(a2, a1)]
    # Default average distance for unlisted pairs
    return 100

print("[*] Initializing RDKit 20 Amino Acid Chemical Matrix...")
AA_PROPS = {}
for code, smi in AA_SMILES.items():
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
        AA_PROPS[code] = {
            'logp': float(Descriptors.MolLogP(mol)),
            'tpsa': float(Descriptors.TPSA(mol)),
            'mw': float(Descriptors.ExactMolWt(mol)),
            'charge': AA_CHARGE.get(code, 0.0),
            'fp': fp
        }

print("[+] RDKit Amino Acid Engine initialized successfully.")

# Load Gene Strands & Lengths from Reference Panel FNA
def load_gene_metadata(fna_path=REF_PANEL_FNA):
    gene_meta = {}
    if not os.path.exists(fna_path):
        print(f"[!] FNA path not found: {fna_path}")
        return gene_meta
        
    with open(fna_path, "r") as f:
        for line in f:
            if line.startswith(">"):
                parts = line.strip()[1:].split()
                gene = parts[0]
                meta = {'strand': '+', 'length': 50000}
                for p in parts[1:]:
                    if "=" in p:
                        k, v = p.split("=", 1)
                        if k == "strand": meta['strand'] = v
                        elif k == "length": 
                            try: meta['length'] = int(v.replace("bp", ""))
                            except: pass
                gene_meta[gene] = meta
    return gene_meta

GENE_METADATA = load_gene_metadata()
print(f"[+] Loaded metadata for {len(GENE_METADATA)} genes from reference panel.")

def compute_mutation_features(row):
    chrom = str(row.get('chrom', ''))
    pos = int(row.get('pos', 1))
    ref = str(row.get('ref', 'N')).upper()
    alt = str(row.get('alt', 'N')).upper()
    ref_seq = str(row.get('ref_seq_21', 'N'*21)).upper()
    alt_seq = str(row.get('alt_seq_21', 'N'*21)).upper()
    trinuc = str(row.get('trinucleotide', 'N[N>N]N'))
    
    meta = GENE_METADATA.get(chrom, {'strand': '+', 'length': 50000})
    strand = meta['strand']
    gene_len = meta['length']
    
    # 1. Determine Codon Phase: (pos - 1) % 3
    phase = (pos - 1) % 3
    if phase == 0:
        ref_codon = ref_seq[10:13] if len(ref_seq) >= 13 else 'ATG'
        alt_codon = alt_seq[10:13] if len(alt_seq) >= 13 else 'ATG'
    elif phase == 1:
        ref_codon = ref_seq[9:12] if len(ref_seq) >= 12 else 'ATG'
        alt_codon = alt_seq[9:12] if len(alt_seq) >= 12 else 'ATG'
    else:
        ref_codon = ref_seq[8:11] if len(ref_seq) >= 11 else 'ATG'
        alt_codon = alt_seq[8:11] if len(alt_seq) >= 11 else 'ATG'
        
    # If gene is on negative strand, reverse complement codons
    if strand == '-':
        ref_codon = rev_comp(ref_codon)
        alt_codon = rev_comp(alt_codon)
        
    ref_aa = GENETIC_CODE.get(ref_codon, 'X')
    alt_aa = GENETIC_CODE.get(alt_codon, 'X')
    is_syn = 1 if ref_aa == alt_aa else 0
    
    # RDKit Chemistry Properties
    ref_p = AA_PROPS.get(ref_aa, AA_PROPS['A'])
    alt_p = AA_PROPS.get(alt_aa, AA_PROPS['A'])
    
    delta_logp = round(alt_p['logp'] - ref_p['logp'], 3)
    delta_tpsa = round(alt_p['tpsa'] - ref_p['tpsa'], 3)
    delta_mw = round(alt_p['mw'] - ref_p['mw'], 3)
    delta_charge = round(alt_p['charge'] - ref_p['charge'], 1)
    
    sim = DataStructs.TanimotoSimilarity(ref_p['fp'], alt_p['fp'])
    tanimoto_chem_dist = round(1.0 - sim, 3)
    grantham = get_grantham(ref_aa, alt_aa)
    
    # 2. Evolutionary Selection (dN, dS)
    # Estimated gene-level non-synonymous (N) and synonymous (S) sites
    pot_N = max(100.0, gene_len * 0.75)
    pot_S = max(30.0, gene_len * 0.25)
    
    if is_syn == 1:
        nd_count = 0.0
        sd_count = 1.0
    else:
        nd_count = 1.0
        sd_count = 0.0
        
    # Nei-Gojobori one-step rate per locus
    dn = round(nd_count / pot_N, 6)
    ds = round(sd_count / pot_S, 6)
    dn_ds_ratio = round((dn / (ds + 1e-6)), 3)
    if is_syn == 0 and ds == 0:
        dn_ds_ratio = min(9.999, round(dn * pot_S, 3))
        
    # 3. Mutual Information (MI)
    # mi_window_score: Shannon entropy co-occurrence in 21bp window
    counts = [ref_seq.count(b) for b in ['A', 'C', 'G', 'T']]
    total = sum(counts)
    if total > 0:
        p_vec = [c / total for c in counts if c > 0]
        entropy = -sum(p * math.log2(p) for p in p_vec)
        # Normalized window dependency (max entropy for 4 bases is 2.0 bits)
        mi_window_score = round(1.0 - (entropy / 2.0), 3)
    else:
        mi_window_score = 0.5
        
    # mi_cancer_class_score: High for lung-cancer signature motifs (e.g. C>A transversions / tobacco signature SBS4)
    if ">A" in trinuc or ">T" in trinuc:
        # Classical tobacco smoke transversions / CpG deamination
        mi_cancer_class_score = 0.850 if ">A" in trinuc else 0.650
    elif ">G" in trinuc:
        mi_cancer_class_score = 0.450
    else:
        mi_cancer_class_score = 0.350
        
    return {
        'ref_aa': ref_aa,
        'alt_aa': alt_aa,
        'is_synonymous': is_syn,
        'delta_logp': delta_logp,
        'delta_tpsa': delta_tpsa,
        'delta_mw': delta_mw,
        'delta_charge': delta_charge,
        'tanimoto_chem_dist': tanimoto_chem_dist,
        'grantham_score': grantham,
        'dn': dn,
        'ds': ds,
        'dn_ds_ratio': dn_ds_ratio,
        'mi_window_score': mi_window_score,
        'mi_cancer_class_score': mi_cancer_class_score
    }

print("[+] Bio-Chemistry Engine functions defined and ready!")
