import os
import sys
import gzip
import re
import time
import pandas as pd
import numpy as np

# Import our RDKit biochemistry engine
import bio_chemistry_engine as bce
import rdkit.DataStructs as ds

AA_3_TO_1 = {
    'Ala': 'A', 'Arg': 'R', 'Asn': 'N', 'Asp': 'D', 'Cys': 'C',
    'Gln': 'Q', 'Glu': 'E', 'Gly': 'G', 'His': 'H', 'Ile': 'I',
    'Leu': 'L', 'Lys': 'K', 'Met': 'M', 'Phe': 'F', 'Pro': 'P',
    'Ser': 'S', 'Thr': 'T', 'Trp': 'W', 'Tyr': 'Y', 'Val': 'V',
    'Ter': '*', 'Sec': 'U', '=': '='
}

TRANSITIONS = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}

# 80-128 Target Cancer Predisposition Genes
CANCER_GENES = {
    # HBOC
    'BRCA1': ('Breast_and_Ovarian_Cancer', 1863),
    'BRCA2': ('Breast_and_Ovarian_Cancer', 3418),
    'PALB2': ('Breast_and_Ovarian_Cancer', 1186),
    'ATM': ('Breast_and_Ovarian_Cancer', 3056),
    'CHEK2': ('Breast_and_Ovarian_Cancer', 543),
    'BARD1': ('Breast_and_Ovarian_Cancer', 777),
    'BRIP1': ('Breast_and_Ovarian_Cancer', 1249),
    'RAD51C': ('Breast_and_Ovarian_Cancer', 376),
    'RAD51D': ('Breast_and_Ovarian_Cancer', 328),
    'RAD51B': ('Breast_and_Ovarian_Cancer', 384),
    'RAD50': ('Breast_and_Ovarian_Cancer', 1312),
    'NBN': ('Breast_and_Ovarian_Cancer', 754),
    'MRE11': ('Breast_and_Ovarian_Cancer', 708),
    # Lynch Syndrome & Mismatch Repair
    'MLH1': ('Lynch_Syndrome_Colorectal', 756),
    'MSH2': ('Lynch_Syndrome_Colorectal', 934),
    'MSH6': ('Lynch_Syndrome_Colorectal', 1360),
    'PMS2': ('Lynch_Syndrome_Colorectal', 862),
    'EPCAM': ('Lynch_Syndrome_Colorectal', 314),
    'MLH3': ('Lynch_Syndrome_Colorectal', 1453),
    'MSH3': ('Lynch_Syndrome_Colorectal', 1137),
    'PMS1': ('Lynch_Syndrome_Colorectal', 932),
    # Polyposis
    'APC': ('Polyposis_Colorectal', 2843),
    'MUTYH': ('Polyposis_Colorectal', 535),
    'NTHL1': ('Polyposis_Colorectal', 312),
    'POLE': ('Polyposis_Colorectal', 2286),
    'POLD1': ('Polyposis_Colorectal', 1107),
    'AXIN2': ('Polyposis_Colorectal', 843),
    'GREM1': ('Polyposis_Colorectal', 184),
    'BMPR1A': ('Polyposis_Colorectal', 532),
    'SMAD4': ('Polyposis_Colorectal', 552),
    'RNF43': ('Polyposis_Colorectal', 783),
    # Li-Fraumeni & Cell Cycle
    'TP53': ('Li_Fraumeni_Multi_Organ', 393),
    'RB1': ('Li_Fraumeni_Multi_Organ', 928),
    'CDKN2A': ('Melanoma_Pancreatic_Cancer', 156),
    'CDKN1B': ('Melanoma_Pancreatic_Cancer', 198),
    'CDKN2B': ('Melanoma_Pancreatic_Cancer', 138),
    'CDK4': ('Melanoma_Pancreatic_Cancer', 303),
    'CCND1': ('Li_Fraumeni_Multi_Organ', 295),
    'MDM2': ('Li_Fraumeni_Multi_Organ', 491),
    'MDM4': ('Li_Fraumeni_Multi_Organ', 490),
    'CHEK1': ('Li_Fraumeni_Multi_Organ', 476),
    # Cowden & Hamartoma
    'PTEN': ('Cowden_Syndrome', 403),
    'STK11': ('Melanoma_Pancreatic_Cancer', 433),
    'ENG': ('Cowden_Syndrome', 658),
    # Gastric
    'CDH1': ('Gastric_Cancer', 882),
    'CTNNA1': ('Gastric_Cancer', 906),
    # Endocrine & MEN
    'RET': ('Endocrine_Thyroid_MEN', 1114),
    'MEN1': ('Endocrine_Thyroid_MEN', 610),
    'MAX': ('Endocrine_Thyroid_MEN', 160),
    'TMEM127': ('Endocrine_Thyroid_MEN', 238),
    'SDHA': ('Endocrine_Thyroid_MEN', 664),
    'SDHB': ('Endocrine_Thyroid_MEN', 280),
    'SDHC': ('Endocrine_Thyroid_MEN', 169),
    'SDHD': ('Endocrine_Thyroid_MEN', 159),
    # Renal & Neurofibromatosis
    'VHL': ('Renal_VHL_Syndrome', 213),
    'MET': ('Renal_VHL_Syndrome', 1390),
    'FH': ('Renal_VHL_Syndrome', 510),
    'BAP1': ('Renal_VHL_Syndrome', 729),
    'WT1': ('Renal_VHL_Syndrome', 449),
    'NF1': ('Neurofibromatosis', 2818),
    'NF2': ('Neurofibromatosis', 595),
    'TSC1': ('Tuberous_Sclerosis', 1164),
    'TSC2': ('Tuberous_Sclerosis', 1807),
    # Prostate & Other
    'HOXB13': ('Prostate_Cancer', 284),
    'AR': ('Prostate_Cancer', 920),
    'ESR1': ('Breast_and_Ovarian_Cancer', 595),
    # Fanconi Anemia
    'FANCA': ('Fanconi_Anemia', 1455),
    'FANCC': ('Fanconi_Anemia', 558),
    'FANCD2': ('Fanconi_Anemia', 1471),
    'FANCE': ('Fanconi_Anemia', 536),
    'FANCF': ('Fanconi_Anemia', 374),
    'FANCG': ('Fanconi_Anemia', 622),
    'FANCL': ('Fanconi_Anemia', 375),
    'FANCM': ('Fanconi_Anemia', 2048),
    'WRN': ('Fanconi_Anemia', 1432),
    'RECQL4': ('Fanconi_Anemia', 1208),
    'XRCC2': ('Breast_and_Ovarian_Cancer', 280),
    'XRCC3': ('Breast_and_Ovarian_Cancer', 346),
    # Oncogenes & Kinases
    'EGFR': ('Oncogene_Lung_Glioblastoma', 1210),
    'ERBB2': ('Oncogene_Breast_Gastric', 1255),
    'ERBB3': ('Oncogene_Breast_Gastric', 1342),
    'KIT': ('Gastrointestinal_Stromal', 976),
    'PDGFRA': ('Gastrointestinal_Stromal', 1089),
    'ALK': ('Oncogene_Lung_Lymphoma', 1620),
    'ROS1': ('Oncogene_Lung_Glioblastoma', 2347),
    'KRAS': ('Oncogene_Pancreatic_Colorectal', 189),
    'NRAS': ('Oncogene_Melanoma_Thyroid', 189),
    'HRAS': ('Oncogene_Bladder_Thyroid', 189),
    'BRAF': ('Oncogene_Melanoma_Colorectal', 766),
    'PIK3CA': ('Oncogene_Breast_Colorectal', 1068),
    'PIK3R1': ('Oncogene_Breast_Colorectal', 724),
    'AKT1': ('Oncogene_Breast_Colorectal', 480),
    'MTOR': ('Oncogene_Renal_Colorectal', 2549),
}

def parse_protein_change(name_str):
    """
    Extracts ref_aa, alt_aa, and aa_pos from HGVS protein strings like:
    NM_000059.4(BRCA2):c.1114A>C (p.Asn372His)
    NM_000546.5(TP53):c.818G>A (p.Arg273His)
    """
    match = re.search(r'\(p\.([A-Z][a-z]{2})(\d+)([A-Z][a-z]{2}|=)\)', name_str)
    if match:
        ref_3, pos_str, alt_3 = match.groups()
        ref_aa = AA_3_TO_1.get(ref_3, 'X')
        alt_aa = ref_aa if alt_3 == '=' else AA_3_TO_1.get(alt_3, 'X')
        return ref_aa, alt_aa, int(pos_str)
    return None, None, None

def build_pan_cancer_dataset(
    clinvar_path=None,
    out_csv=None,
    save_vus=True,
    vus_out_csv=None
):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if clinvar_path is None:
        raw_path = os.path.join(script_dir, "raw_data", "variant_summary.txt.gz")
        clinvar_path = raw_path if os.path.exists(raw_path) else os.path.join(script_dir, "variant_summary.txt.gz")
        
    data_dir = os.path.join(script_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    if out_csv is None:
        out_csv = os.path.join(data_dir, "pan_cancer_germline_dataset.csv")
    if vus_out_csv is None:
        vus_out_csv = os.path.join(data_dir, "pan_cancer_vus_dataset.csv")
        
    print(f"[*] Starting Pan-Cancer Germline Extraction across {len(CANCER_GENES)} Cancer Genes...")
    print(f"[*] Source Database: {clinvar_path}")
    t0 = time.time()
    
    clean_records = []
    vus_records = []
    
    with gzip.open(clinvar_path, 'rt', encoding='utf-8', errors='ignore') as f:
        header = f.readline().strip().split('\t')
        col_map = {name: i for i, name in enumerate(header)}
        
        gene_col = col_map['GeneSymbol']
        assembly_col = col_map['Assembly']
        origin_col = col_map['OriginSimple']
        sig_col = col_map['ClinicalSignificance']
        type_col = col_map['Type']
        name_col = col_map['Name']
        chrom_col = col_map['Chromosome']
        pos_col = col_map['PositionVCF']
        ref_col = col_map['ReferenceAlleleVCF']
        alt_col = col_map['AlternateAlleleVCF']
        
        line_count = 0
        
        for line in f:
            line_count += 1
            parts = line.strip().split('\t')
            if len(parts) <= pos_col:
                continue
            if parts[assembly_col] != 'GRCh38':
                continue
                
            gene = parts[gene_col]
            if gene not in CANCER_GENES:
                continue
                
            origin = parts[origin_col].lower()
            if 'germline' not in origin:
                continue
                
            # Filter for single nucleotide variants
            ref = parts[ref_col].upper()
            alt = parts[alt_col].upper()
            if len(ref) != 1 or len(alt) != 1:
                continue
                
            sig = parts[sig_col].lower()
            label = None
            if 'pathogenic' in sig and 'conflict' not in sig and 'uncertain' not in sig and 'benign' not in sig:
                label = 1 # Cancer Risk
            elif 'benign' in sig and 'conflict' not in sig and 'uncertain' not in sig and 'pathogenic' not in sig:
                label = 0 # Safe Control
            elif ('uncertain' in sig or 'vus' in sig) and 'conflict' not in sig:
                label = 2 # VUS
            else:
                continue
                
            name = parts[name_col]
            chrom = parts[chrom_col]
            try:
                pos = int(parts[pos_col])
            except ValueError:
                continue
                
            syndrome, max_aa = CANCER_GENES[gene]
            
            ref_aa, alt_aa, aa_pos = parse_protein_change(name)
            is_missense = 1 if (ref_aa and alt_aa and ref_aa != alt_aa and ref_aa != '*' and alt_aa != '*') else 0
            is_nonsense = 1 if (alt_aa == '*') else 0
            is_synonymous = 1 if (ref_aa and alt_aa and ref_aa == alt_aa) else 0
            is_transition = 1 if (ref, alt) in TRANSITIONS else 0
            
            # Biophysical Chemistry via RDKit
            ref_p = bce.AA_PROPS.get(ref_aa, bce.AA_PROPS['A']) if ref_aa else bce.AA_PROPS['A']
            alt_p = bce.AA_PROPS.get(alt_aa, bce.AA_PROPS['A']) if alt_aa else bce.AA_PROPS['A']
            
            delta_mw = round(alt_p['mw'] - ref_p['mw'], 3) if ref_aa and alt_aa else 0.0
            delta_logp = round(alt_p['logp'] - ref_p['logp'], 3) if ref_aa and alt_aa else 0.0
            delta_tpsa = round(alt_p['tpsa'] - ref_p['tpsa'], 3) if ref_aa and alt_aa else 0.0
            delta_charge = round(alt_p['charge'] - ref_p['charge'], 1) if ref_aa and alt_aa else 0.0
            grantham = bce.get_grantham(ref_aa, alt_aa) if ref_aa and alt_aa else 0
            
            sim = ds.TanimotoSimilarity(ref_p['fp'], alt_p['fp']) if ref_aa and alt_aa else 1.0
            tanimoto_chem_dist = round(1.0 - sim, 3)
            
            rel_protein_pos = round(aa_pos / max_aa, 4) if aa_pos else 0.5
            
            row = {
                'gene': gene,
                'chrom': chrom,
                'pos': pos,
                'ref': ref,
                'alt': alt,
                'ref_aa': ref_aa if ref_aa else '.',
                'alt_aa': alt_aa if alt_aa else '.',
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
                'cancer_syndrome': syndrome,
                'label': label
            }
            
            if label in (0, 1):
                clean_records.append(row)
            elif label == 2:
                vus_records.append(row)
                
    elapsed = time.time() - t0
    print(f"[+] Streaming completed in {elapsed:.1f} seconds!")
    print(f"[+] Total Clean Labeled Variants extracted: {len(clean_records):,}")
    
    df_clean = pd.DataFrame(clean_records)
    print(f"[*] Label distribution in Training Set:")
    print(df_clean['label'].value_counts())
    print(f"[*] Top Genes extracted:")
    print(df_clean['gene'].value_counts().head(10))
    
    print(f"[*] Saving Clean Pan-Cancer Dataset to {out_csv}...")
    df_clean.to_csv(out_csv, index=False)
    print(f"[+] Successfully wrote {out_csv} ({os.path.getsize(out_csv) / (1024*1024):.2f} MB)")
    
    if save_vus and vus_records:
        df_vus = pd.DataFrame(vus_records)
        print(f"[*] Saving VUS Dataset to {vus_out_csv} ({len(vus_records):,} variants)...")
        df_vus.to_csv(vus_out_csv, index=False)
        print(f"[+] Successfully wrote {vus_out_csv} ({os.path.getsize(vus_out_csv) / (1024*1024):.2f} MB)")

if __name__ == '__main__':
    build_pan_cancer_dataset()
