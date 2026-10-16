import os
import sys
import gzip
import re
import pandas as pd
import numpy as np

# Import our RDKit biochemistry engine
import bio_chemistry_engine as bce

AA_3_TO_1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*', 'Sec': 'U', '=': '='
}

TRANSITIONS = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}

def parse_protein_change(name_str):
    """
    Extracts ref_aa, alt_aa, and aa_pos from HGVS protein strings like:
    NM_000059.4(BRCA2):c.1114A>C (p.Asn372His)
    NM_000059.4(BRCA2):c.8219T>A (p.Leu2740Ter)
    """
    match = re.search(r'\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=)\)', name_str)
    if match:
        ref_3, pos_str, alt_3 = match.groups()
        ref_aa = AA_3_TO_1.get(ref_3, 'X')
        alt_aa = ref_aa if alt_3 == '=' else AA_3_TO_1.get(alt_3, 'X')
        return ref_aa, alt_aa, int(pos_str)
    return None, None, None

def build_pilot_dataset(clinvar_path="variant_summary.txt.gz", out_csv="brca_germline_pilot_dataset.csv"):
    print(f"[*] Streaming ClinVar: {clinvar_path}...")
    records = []
    
    with gzip.open(clinvar_path, 'rt', encoding='utf-8') as f:
        header = f.readline().strip().split('\t')
        col_map = {name: i for i, name in enumerate(header)}
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) <= col_map['Assembly']: continue
            if parts[col_map['Assembly']] != 'GRCh38': continue
            
            gene = parts[col_map['GeneSymbol']]
            if gene not in ('BRCA1', 'BRCA2'): continue
            if 'germline' not in parts[col_map['OriginSimple']].lower(): continue
            if parts[col_map['Type']] != 'single nucleotide variant': continue
            
            sig = parts[col_map['ClinicalSignificance']].lower()
            if 'pathogenic' in sig and 'conflicting' not in sig and 'uncertain' not in sig and 'benign' not in sig:
                label = 1 # High Cancer Risk
            elif 'benign' in sig and 'conflicting' not in sig and 'uncertain' not in sig and 'pathogenic' not in sig:
                label = 0 # Safe / Benign Control
            else:
                continue
                
            name = parts[col_map['Name']]
            chrom = parts[col_map['Chromosome']]
            pos = int(parts[col_map['PositionVCF']])
            ref = parts[col_map['ReferenceAlleleVCF']].upper()
            alt = parts[col_map['AlternateAlleleVCF']].upper()
            review_status = parts[col_map['ReviewStatus']]
            phenotypes = parts[col_map['PhenotypeList']]
            
            # Keep standard single base SNVs
            if len(ref) != 1 or len(alt) != 1:
                continue
                
            ref_aa, alt_aa, aa_pos = parse_protein_change(name)
            is_missense = 1 if (ref_aa and alt_aa and ref_aa != alt_aa and ref_aa != '*' and alt_aa != '*') else 0
            is_nonsense = 1 if (alt_aa == '*') else 0
            is_synonymous = 1 if (ref_aa and alt_aa and ref_aa == alt_aa) else 0
            
            # Transition vs Transversion
            is_transition = 1 if (ref, alt) in TRANSITIONS else 0
            
            # Chemistry deltas via RDKit engine
            ref_p = bce.AA_PROPS.get(ref_aa, bce.AA_PROPS['A']) if ref_aa else bce.AA_PROPS['A']
            alt_p = bce.AA_PROPS.get(alt_aa, bce.AA_PROPS['A']) if alt_aa else bce.AA_PROPS['A']
            
            delta_mw = round(alt_p['mw'] - ref_p['mw'], 3) if ref_aa and alt_aa else 0.0
            delta_logp = round(alt_p['logp'] - ref_p['logp'], 3) if ref_aa and alt_aa else 0.0
            delta_tpsa = round(alt_p['tpsa'] - ref_p['tpsa'], 3) if ref_aa and alt_aa else 0.0
            delta_charge = round(alt_p['charge'] - ref_p['charge'], 1) if ref_aa and alt_aa else 0.0
            grantham = bce.get_grantham(ref_aa, alt_aa) if ref_aa and alt_aa else 0
            
            import rdkit.DataStructs as ds
            sim = ds.TanimotoSimilarity(ref_p['fp'], alt_p['fp']) if ref_aa and alt_aa else 1.0
            tanimoto_chem_dist = round(1.0 - sim, 3)
            
            # Gene specific relative protein position
            # BRCA1 has 1863 amino acids, BRCA2 has 3418 amino acids
            max_aa = 1863 if gene == 'BRCA1' else 3418
            rel_protein_pos = round(aa_pos / max_aa, 4) if aa_pos else 0.5
            
            records.append({
                'gene': gene,
                'chrom': chrom,
                'pos': pos,
                'ref': ref,
                'alt': alt,
                'ref_aa': ref_aa if ref_aa else 'None',
                'alt_aa': alt_aa if alt_aa else 'None',
                'aa_pos': aa_pos if aa_pos else 0,
                'rel_protein_pos': rel_protein_pos,
                'is_transition': is_transition,
                'is_missense': is_missense,
                'is_nonsense': is_nonsense,
                'is_synonymous': is_synonymous,
                'delta_mw': delta_mw,
                'delta_logp': delta_logp,
                'delta_tpsa': delta_tpsa,
                'delta_charge': delta_charge,
                'tanimoto_chem_dist': tanimoto_chem_dist,
                'grantham_score': grantham,
                'cancer_target': 'Breast_and_Ovarian_Cancer',
                'label': label  # 1 = Pathogenic, 0 = Benign
            })
            
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False)
    print(f"[+] Saved clean pilot dataset: {out_csv}")
    print(f"    Total Variants: {len(df):,}")
    print(f"    Pathogenic (Risk=1): {(df['label']==1).sum():,}")
    print(f"    Benign (Safe=0):     {(df['label']==0).sum():,}")
    print(f"    BRCA1: {(df['gene']=='BRCA1').sum():,}, BRCA2: {(df['gene']=='BRCA2').sum():,}")
    return df

if __name__ == '__main__':
    build_pilot_dataset()
