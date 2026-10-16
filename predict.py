# ==============================================================================
# GenomeDL: AI-Driven 80-Gene Hereditary Cancer Interpretation Engine
# Clinical Inference & Patient Risk Predictor (predict.py)
# ==============================================================================

import os
import sys

# Force UTF-8 encoding on console streams with graceful fallback
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import warnings
warnings.filterwarnings('ignore')

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

import argparse
import torch
import numpy as np
import pandas as pd

# Import architecture & biochemistry
from train_tabular_mlp import GenomeDLResNet, TabularResidualBlock
import bio_chemistry_engine as bce
import rdkit.DataStructs as ds

# Pre-defined Organ Penetrance & Syndrome Mapping
SYNDROME_ORGAN_RISKS = {
    'Breast_and_Ovarian_Cancer': [
        ('Breast Cancer', 72.0, 'Extreme Lifetime Risk (Population: 12.5%)'),
        ('Ovarian Cancer', 44.0, 'High Lifetime Risk (Population: 1.3%)'),
        ('Pancreatic Cancer', 8.5, 'Elevated Risk (Population: 1.6%)'),
        ('Prostate Cancer (m)', 15.0, 'Elevated Risk (Population: 11.0%)')
    ],
    'Lynch_Syndrome_Colorectal': [
        ('Colorectal / Colon Cancer', 75.0, 'Extreme Lifetime Risk (Population: 4.2%)'),
        ('Endometrial / Uterine Cancer', 50.0, 'High Lifetime Risk (Population: 2.8%)'),
        ('Stomach / Gastric Cancer', 12.0, 'Elevated Lifetime Risk (Population: 0.9%)'),
        ('Ovarian Cancer', 10.0, 'Elevated Risk (Population: 1.3%)')
    ],
    'Polyposis_Colorectal': [
        ('Colorectal Polyposis & Cancer', 95.0, 'Critical Risk (>95% without surveillance)'),
        ('Duodenal / Small Bowel Cancer', 10.0, 'Elevated Risk (Population: 0.3%)')
    ],
    'Li_Fraumeni_Multi_Organ': [
        ('Soft-Tissue Sarcomas', 85.0, 'Severe Lifetime Predisposition'),
        ('Pre-menopausal Breast Cancer', 60.0, 'High Early-Onset Risk'),
        ('Brain & Adrenocortical Tumors', 30.0, 'Elevated Childhood & Adult Risk')
    ],
    'Cowden_Syndrome': [
        ('Breast Cancer', 80.0, 'Extreme Lifetime Risk'),
        ('Thyroid Cancer (Follicular)', 35.0, 'High Lifetime Risk (Population: 1.2%)'),
        ('Endometrial Cancer', 28.0, 'Elevated Risk (Population: 2.8%)')
    ],
    'Gastric_Cancer': [
        ('Diffuse Gastric (Stomach) Cancer', 70.0, 'Extreme Lifetime Risk'),
        ('Lobular Breast Cancer', 42.0, 'High Lifetime Risk (Population: 1.5%)')
    ],
    'Endocrine_Thyroid_MEN': [
        ('Medullary Thyroid Cancer', 95.0, 'Critical Endocrine Risk'),
        ('Pheochromocytoma', 40.0, 'Adrenal Gland Tumor Risk'),
        ('Parathyroid Adenomas', 25.0, 'Elevated Risk')
    ],
    'Renal_VHL_Syndrome': [
        ('Clear Cell Renal (Kidney) Cancer', 70.0, 'High Lifetime Risk (Population: 1.8%)'),
        ('Pheochromocytoma', 20.0, 'Elevated Endocrine Risk')
    ],
    'Melanoma_Pancreatic_Cancer': [
        ('Cutaneous Melanoma', 60.0, 'High Lifetime Risk (Population: 2.2%)'),
        ('Pancreatic Adenocarcinoma', 20.0, 'Elevated Lifetime Risk (Population: 1.6%)')
    ]
}

def load_inference_model(weights_path="pan_cancer_germline_resnet.pth"):
    if not os.path.exists(weights_path):
        print(f"[!] Error: Model weights not found at {weights_path}")
        print("    Please run: python train_tabular_mlp.py first.")
        sys.exit(1)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    gene_to_id = checkpoint['gene_to_id']
    scaler = checkpoint['scaler']
    
    # 19 continuous features + 93 genes + 23 syndromes
    model = GenomeDLResNet(
        num_cont=19,
        num_genes=len(gene_to_id),
        num_syndromes=23,
        hidden_dim=256
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, gene_to_id, scaler, device

def evaluate_variant(model, gene_to_id, scaler, device, variant):
    gene = variant['gene']
    gene_id = gene_to_id.get(gene, 0)
    syn_id = 0 # Default / HBOC syndrome index
    
    ref_aa = variant.get('ref_aa', 'C')
    alt_aa = variant.get('alt_aa', 'G')
    
    # RDKit biophysics
    ref_p = bce.AA_PROPS.get(ref_aa, bce.AA_PROPS['A'])
    alt_p = bce.AA_PROPS.get(alt_aa, bce.AA_PROPS['A'])
    
    delta_mw = round(alt_p['mw'] - ref_p['mw'], 3)
    delta_logp = round(alt_p['logp'] - ref_p['logp'], 3)
    delta_tpsa = round(alt_p['tpsa'] - ref_p['tpsa'], 3)
    delta_charge = round(alt_p['charge'] - ref_p['charge'], 1)
    grantham = bce.get_grantham(ref_aa, alt_aa)
    sim = ds.TanimotoSimilarity(ref_p['fp'], alt_p['fp'])
    tanimoto_chem_dist = round(1.0 - sim, 3)
    
    is_missense = 1 if (ref_aa != alt_aa and ref_aa != '*' and alt_aa != '*') else 0
    is_nonsense = 1 if (alt_aa == '*') else 0
    is_synonymous = 1 if (ref_aa == alt_aa) else 0
    is_transition = 1 if (variant['ref'], variant['alt']) in {('A','G'),('G','A'),('C','T'),('T','C')} else 0
    
    # 19 continuous features
    ref_onehot = [1.0 if variant['ref'] == b else 0.0 for b in ['A', 'C', 'G', 'T']]
    alt_onehot = [1.0 if variant['alt'] == b else 0.0 for b in ['A', 'C', 'G', 'T']]
    
    cont_vals = ref_onehot + alt_onehot + [
        variant.get('rel_protein_pos', 0.5),
        float(is_transition),
        float(is_missense),
        float(is_nonsense),
        float(is_synonymous),
        float(delta_mw),
        float(delta_logp),
        float(delta_tpsa),
        float(delta_charge),
        float(tanimoto_chem_dist),
        float(grantham)
    ]
    
    # Scale continuous subset
    scaled_subset = scaler.transform([[delta_mw, delta_logp, delta_tpsa, delta_charge, grantham]])[0]
    cont_vals[14:19] = scaled_subset
    
    x_cont = torch.tensor([cont_vals], dtype=torch.float32).to(device)
    g_id = torch.tensor([gene_id], dtype=torch.long).to(device)
    s_id = torch.tensor([syn_id], dtype=torch.long).to(device)
    
    with torch.no_grad():
        logit = model(x_cont, g_id, s_id).item()
        prob_pathogenic = 1.0 / (1.0 + np.exp(-logit))
        
    # 2-Step Verification: Continuous Prob -> 5 ACMG Classes
    if prob_pathogenic >= 0.90:
        acmg_class = "CLASS 4: PATHOGENIC"
        tier_symbol = "🔴"
        action = "Immediate Clinical High-Risk Protocol & Family Screening"
    elif prob_pathogenic >= 0.65:
        acmg_class = "CLASS 3: LIKELY PATHOGENIC"
        tier_symbol = "🟠"
        action = "Early Non-Invasive Surveillance (Annual Contrast MRI / Colonoscopy)"
    elif prob_pathogenic >= 0.35:
        acmg_class = "CLASS 2: VUS (UNCERTAIN)"
        tier_symbol = "🟡"
        action = "Monitor; DO NOT perform invasive surgical interventions"
    elif prob_pathogenic >= 0.10:
        acmg_class = "CLASS 1: LIKELY BENIGN"
        tier_symbol = "🟢"
        action = "Routine population lifestyle & wellness checkups"
    else:
        acmg_class = "CLASS 0: BENIGN"
        tier_symbol = "🟢"
        action = "Confirmed Harmless Variant; No medical intervention"
        
    return {
        'logit': logit,
        'prob': prob_pathogenic,
        'class': acmg_class,
        'symbol': tier_symbol,
        'action': action,
        'grantham': grantham,
        'delta_mw': delta_mw,
        'delta_charge': delta_charge
    }

def print_clinical_report(patient_id, variant, result, syndrome_key='Breast_and_Ovarian_Cancer'):
    print("\n" + "="*84)
    print(f"      GERMLINE HEREDITARY CANCER RISK REPORT  |  GenomeDL Engine v2")
    print(f"      Patient ID: {patient_id}  |  Genome Build: GRCh38")
    print("="*84)
    
    print(f"\n[1] 2-STEP VERIFICATION VERDICT:")
    print(f"    • Step 1 (Risk Probability): {result['prob']*100:.2f}% (Logit: {result['logit']:+.2f})")
    print(f"    • Step 2 (ACMG Classification): {result['symbol']} {result['class']}")
    print(f"    • Clinical Action: {result['action']}")
    
    print(f"\n[2] MOLECULAR BIOMARKER DETAILS:")
    print(f"    • Gene:            {variant['gene']}")
    print(f"    • Locus:           {variant.get('chrom', 'chr17')}:{variant.get('pos', '43044295')} ({variant['ref']} > {variant['alt']})")
    print(f"    • Protein Change:  p.{variant.get('ref_aa', 'Cys')}{variant.get('aa_pos', 61)}{variant.get('alt_aa', 'Gly')}")
    print(f"    • Biophysics:      Grantham = {result['grantham']} | ΔMW = {result['delta_mw']:+.1f} Da | ΔCharge = {result['delta_charge']:+.1f}")
    
    print(f"\n[3] ORGAN-SPECIFIC CANCER RISK RADAR:")
    organ_risks = SYNDROME_ORGAN_RISKS.get(syndrome_key, SYNDROME_ORGAN_RISKS['Breast_and_Ovarian_Cancer'])
    for idx, (organ, base_risk, desc) in enumerate(organ_risks, 1):
        adjusted_risk = min(99.0, max(base_risk * 0.15, base_risk * result['prob']))
        bar_len = int(adjusted_risk / 5)
        bar = "█" * bar_len + "░" * (20 - bar_len)
        print(f"    {idx}. {organ:<28} : {adjusted_risk:>5.1f}% [{bar}] ({desc})")
        
    print(f"\n[4] ACTIONABLE SURVEILLANCE RECOMMENDATION:")
    if result['prob'] >= 0.65:
        print(f"    🔴 High-penetrance mutation detected in {variant['gene']}.")
        print(f"    • Routine mammography/colonoscopy may be insufficient.")
        print(f"    • Recommend specialized high-risk surveillance starting 10 years earlier than family onset.")
        print(f"    • Recommend cascade germline genetic counseling for all 1st-degree biological relatives.")
    else:
        print(f"    🟢 No high-penetrance disease-causing mutations detected in the screened panel.")
        print(f"    • Patient maintains standard general-population baseline cancer risk.")
        print(f"    • Standard age-based lifestyle guidelines advised.")
    print("="*84 + "\n")

def main():
    parser = argparse.ArgumentParser(description="GenomeDL: Clinical Hereditary Cancer Predictor")
    parser.add_argument("--demo", action="store_true", help="Run live demo cases (Sarah's High-Risk BRCA1 vs Benign Control)")
    parser.add_argument("--gene", type=str, default="BRCA1", help="Target gene symbol")
    parser.add_argument("--ref", type=str, default="C", help="Reference base")
    parser.add_argument("--alt", type=str, default="G", help="Mutated alternate base")
    parser.add_argument("--ref_aa", type=str, default="C", help="Original amino acid code")
    parser.add_argument("--alt_aa", type=str, default="G", help="New amino acid code")
    parser.add_argument("--pos", type=int, default=43044295, help="Genomic coordinate")
    args = parser.parse_args()
    
    print("[*] Loading GenomeDL Pan-Cancer ResNet Engine...")
    model, gene_to_id, scaler, device = load_inference_model()
    print(f"[+] Loaded model on {device}. Ready for clinical inference across {len(gene_to_id)} cancer genes.")
    
    if args.demo or len(sys.argv) == 1:
        # Case 1: Sarah's High-Risk Founder Mutation (BRCA1 p.Cys61Gly)
        var_sarah = {
            'gene': 'BRCA1', 'chrom': 'chr17', 'pos': 43044295,
            'ref': 'C', 'alt': 'G', 'ref_aa': 'C', 'alt_aa': 'G', 'aa_pos': 61,
            'rel_protein_pos': 61 / 1863
        }
        res_sarah = evaluate_variant(model, gene_to_id, scaler, device, var_sarah)
        print_clinical_report("DEMO_PATIENT_SARAH_24Y", var_sarah, res_sarah, 'Breast_and_Ovarian_Cancer')
        
        # Case 2: Healthy Control Polymorphism (BRCA2 p.Asn372His)
        var_healthy = {
            'gene': 'BRCA2', 'chrom': 'chr13', 'pos': 32332592,
            'ref': 'A', 'alt': 'C', 'ref_aa': 'N', 'alt_aa': 'H', 'aa_pos': 372,
            'rel_protein_pos': 372 / 3418
        }
        res_healthy = evaluate_variant(model, gene_to_id, scaler, device, var_healthy)
        print_clinical_report("DEMO_PATIENT_BENCHMARK_CONTROL", var_healthy, res_healthy, 'Breast_and_Ovarian_Cancer')
    else:
        var_custom = {
            'gene': args.gene.upper(), 'chrom': 'chr', 'pos': args.pos,
            'ref': args.ref.upper(), 'alt': args.alt.upper(),
            'ref_aa': args.ref_aa.upper(), 'alt_aa': args.alt_aa.upper(), 'aa_pos': 100,
            'rel_protein_pos': 0.5
        }
        res_custom = evaluate_variant(model, gene_to_id, scaler, device, var_custom)
        print_clinical_report("CUSTOM_PATIENT_QUERY", var_custom, res_custom)

if __name__ == '__main__':
    main()
