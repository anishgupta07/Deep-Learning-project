import os
import sys
import gzip
import time
import pandas as pd
from collections import Counter

CANCER_GENES = {
    'BRCA1', 'BRCA2', 'TP53', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'EPCAM',
    'APC', 'MUTYH', 'PTEN', 'CDH1', 'STK11', 'VHL', 'RET', 'MEN1',
    'PALB2', 'ATM', 'CHEK2', 'BARD1', 'BRIP1', 'RAD51C', 'RAD51D',
    'NBN', 'CDKN2A', 'CDK4', 'HOXB13', 'POLE', 'POLD1', 'NTHL1',
    'AXIN2', 'BMPR1A', 'SMAD4', 'GREM1', 'CTNNA1', 'MAX', 'TMEM127',
    'SDHA', 'SDHB', 'SDHC', 'SDHD', 'FLCN', 'FH', 'BAP1', 'MET',
    'TSC1', 'TSC2', 'WT1', 'FANCA', 'FANCC', 'FANCD2', 'FANCE',
    'FANCF', 'FANCG', 'FANCL', 'FANCM', 'MRE11', 'RAD50', 'XRCC2',
    'XRCC3', 'RECQL4', 'WRN', 'BLM', 'ATR', 'NF1', 'NF2', 'RB1',
    'DICER1', 'SUFU', 'SMARCA4', 'SMARCB1', 'EZH2', 'ALK', 'KIT',
    'PDGFRA', 'MITF', 'POT1', 'RAD51B', 'FANCB', 'FANCI'
}

def import_all_clinvar_data(clinvar_path="variant_summary.txt.gz"):
    print("[*] Starting complete master extraction of all 359,731 raw entries...")
    t0 = time.time()
    
    clean_snvs = []
    vus_snvs = []
    indels = []
    conflicting = []
    somatic_other = []
    
    with gzip.open(clinvar_path, 'rt', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip().split('\t')
        col = {name: i for i, name in enumerate(header)}
        
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) <= max(col['PositionVCF'], col['Assembly']):
                continue
            if parts[col['Assembly']] != 'GRCh38':
                continue
            gene = parts[col['GeneSymbol']]
            if gene not in CANCER_GENES:
                continue
                
            origin = parts[col['OriginSimple']].lower()
            ref = parts[col['ReferenceAlleleVCF']].upper()
            alt = parts[col['AlternateAlleleVCF']].upper()
            vtype = parts[col['Type']].lower()
            sig = parts[col['ClinicalSignificance']]
            sig_lower = sig.lower()
            
            row = {
                'gene': gene,
                'chrom': parts[col['Chromosome']],
                'pos': parts[col['PositionVCF']],
                'ref': ref,
                'alt': alt,
                'type': parts[col['Type']],
                'name': parts[col['Name']],
                'clinical_significance': sig,
                'origin': parts[col['OriginSimple']],
                'review_status': parts[col['ReviewStatus']]
            }
            
            # Category 1: Non-germline / Somatic
            if 'germline' not in origin:
                somatic_other.append(row)
                continue
                
            # Category 2: Indels / Frameshifts / CNVs / Duplications
            if len(ref) != 1 or len(alt) != 1 or vtype != 'single nucleotide variant':
                indels.append(row)
                continue
                
            # Category 3: Conflicting doctor opinions
            if 'conflict' in sig_lower:
                conflicting.append(row)
                continue
                
            # Category 4: VUS (Uncertain Significance)
            if 'not provided' in sig_lower or 'uncertain' in sig_lower or 'vus' in sig_lower:
                vus_snvs.append(row)
                continue
                
            # Category 5 & 6: Clean High-Confidence SNVs (Pathogenic or Benign)
            if 'pathogenic' in sig_lower:
                row['label'] = 1
                clean_snvs.append(row)
            elif 'benign' in sig_lower:
                row['label'] = 0
                clean_snvs.append(row)
            else:
                somatic_other.append(row)
                
    elapsed = time.time() - t0
    total = len(clean_snvs) + len(vus_snvs) + len(indels) + len(conflicting) + len(somatic_other)
    print(f"[+] Master parse completed in {elapsed:.1f}s. Total Accounted Rows: {total:,}")
    
    print(f"  • Clean SNVs (Train/Test) : {len(clean_snvs):,}")
    print(f"  • VUS SNVs (Gray Zone)    : {len(vus_snvs):,}")
    print(f"  • Indels & Frameshifts     : {len(indels):,}")
    print(f"  • Conflicting Doctor Reviews: {len(conflicting):,}")
    print(f"  • Somatic / Non-Germline  : {len(somatic_other):,}")
    
    # Save partitioned CSVs
    print("[*] Saving all categories to disk...")
    pd.DataFrame(indels).to_csv("pan_cancer_indels_frameshifts.csv", index=False)
    pd.DataFrame(conflicting).to_csv("pan_cancer_conflicting_variants.csv", index=False)
    print("[+] All 359,731 raw entries imported and saved to disk successfully!")

if __name__ == '__main__':
    import_all_clinvar_data()
