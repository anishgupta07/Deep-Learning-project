#!/usr/bin/env python3
"""
==============================================================================
GenomeDL v2: Clinical Single-Variant Inference & Pathogenicity Predictor
==============================================================================
Evaluates individual genetic variants in any of the 130 hereditary cancer genes
using the calibrated GenomeDL v2 Deep Tabular Residual Network.
"""

import os
import sys
import argparse
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import numpy as np
import pandas as pd

from train_v2_model import GenomeDLResNetV2, FEATURE_COLS, SCALE_COLS
from build_features_v2 import build_pairwise_biophysical_lookup, get_protein_lengths_dict, TRANSITIONS

def load_v2_model(weights_path=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if weights_path is None:
        v2_path = os.path.join(script_dir, "models", "v2_pan_cancer_resnet.pth")
        weights_path = v2_path if os.path.exists(v2_path) else os.path.join(script_dir, "models", "pan_cancer_germline_resnet.pth")
        
    if not os.path.exists(weights_path):
        sys.exit(f"[!] Error: Model weights not found at {weights_path}")
        
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    gene_to_id = checkpoint['gene_to_id']
    syn_to_id = checkpoint.get('syndrome_to_id', {})
    scaler = checkpoint['scaler']
    temperature = checkpoint.get('temperature', 0.8769)
    
    model = GenomeDLResNetV2(
        num_cont=len(FEATURE_COLS),
        num_genes=len(gene_to_id),
        num_syndromes=len(syn_to_id),
        hidden_dim=256
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, gene_to_id, syn_to_id, scaler, temperature, device

# Gene to Syndrome default lookup for 130 genes
from extract_clinvar_hereditary import GENE2SYN

def featurize_single_variant(variant, pairwise_lookup, prot_lens):
    gene = variant['gene']
    ref = variant['ref'].upper()
    alt = variant['alt'].upper()
    ref_aa = variant.get('ref_aa', '.').upper()
    alt_aa = variant.get('alt_aa', '.').upper()
    aa_pos = int(variant.get('aa_pos', 0))
    vclass = variant.get('variant_class', 'missense' if ref_aa != alt_aa and ref_aa != '.' else 'intronic')
    
    # 1. Base one-hots
    ref_onehot = [1.0 if ref == b else 0.0 for b in ['A', 'C', 'G', 'T']]
    alt_onehot = [1.0 if alt == b else 0.0 for b in ['A', 'C', 'G', 'T']]
    is_trans = 1.0 if (ref, alt) in TRANSITIONS else 0.0
    
    # 2. Molecular class flags
    is_miss = 1.0 if vclass == 'missense' else 0.0
    is_non = 1.0 if vclass == 'nonsense' or alt_aa == '*' else 0.0
    is_syn = 1.0 if vclass == 'synonymous' or (ref_aa == alt_aa and ref_aa != '.') else 0.0
    is_sl = 1.0 if vclass == 'start_lost' or (ref_aa == 'M' and aa_pos == 1 and alt_aa != 'M') else 0.0
    is_sc = 1.0 if vclass == 'splice_canonical' else 0.0
    is_sr = 1.0 if vclass == 'splice_region' else 0.0
    is_int = 1.0 if vclass == 'intronic' else 0.0
    is_utr = 1.0 if vclass in ('utr5', 'utr3') else 0.0
    is_onc = 1.0 if vclass in ('other_noncoding', 'coding_no_protein_annot') else 0.0
    
    # 3. Position metrics
    g_len = prot_lens.get(gene, 1000)
    if is_miss or is_non or is_syn or is_sl:
        rel_pos = min(1.0, max(0.0, float(aa_pos) / float(g_len)))
        has_prot = 1.0
    else:
        rel_pos = 0.0
        has_prot = 0.0
        
    # 4. Biophysical features
    if (ref_aa, alt_aa) in pairwise_lookup:
        (g_dist, b_score, d_mw, d_logp, d_tpsa, d_chg, c_dist, pol_chg, chg_chg) = pairwise_lookup[(ref_aa, alt_aa)]
    elif is_non:
        g_dist, b_score, d_mw, d_logp, d_tpsa, d_chg, c_dist, pol_chg, chg_chg = (200.0, -4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
    elif is_syn:
        g_dist, b_score, d_mw, d_logp, d_tpsa, d_chg, c_dist, pol_chg, chg_chg = (0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    else:
        g_dist, b_score, d_mw, d_logp, d_tpsa, d_chg, c_dist, pol_chg, chg_chg = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        
    feat_vector = (
        ref_onehot + alt_onehot + [is_trans] +
        [is_miss, is_non, is_syn, is_sl, is_sc, is_sr, is_int, is_utr, is_onc] +
        [has_prot, rel_pos, g_dist, b_score, d_mw, d_logp, d_tpsa, d_chg, c_dist, pol_chg, chg_chg]
    )
    return np.array(feat_vector, dtype=np.float32), g_dist, d_mw, d_chg

def predict_variant_pathogenicity(model, gene_to_id, syn_to_id, scaler, temperature, device, variant, pairwise_lookup, prot_lens):
    gene = variant['gene'].upper()
    if gene not in gene_to_id:
        print(f"[!] Warning: Gene {gene} not in 130-gene training panel. Falling back to index 0.")
        gene_id = 0
    else:
        gene_id = gene_to_id[gene]
        
    syndrome = GENE2SYN.get(gene, 'Other_hereditary')
    syn_id = syn_to_id.get(syndrome, 0)
    
    feat_vec, grantham, delta_mw, delta_charge = featurize_single_variant(variant, pairwise_lookup, prot_lens)
    
    # Scale continuous subset
    sub_idx = [FEATURE_COLS.index(c) for c in SCALE_COLS]
    feat_vec[sub_idx] = scaler.transform([feat_vec[sub_idx]])[0]
    
    gt = torch.tensor([gene_id], dtype=torch.long).to(device)
    st = torch.tensor([syn_id], dtype=torch.long).to(device)
    xt = torch.tensor(np.array([feat_vec]), dtype=torch.float32).to(device)
    
    with torch.no_grad():
        logit = model(xt, gt, st).item()
        prob = 1.0 / (1.0 + np.exp(-logit / temperature))
        
    if prob >= 0.85:
        acmg_class = "CLASS 4: PATHOGENIC"
        symbol = "[!]"
        action = "Immediate Clinical High-Risk Protocol & Cascade Family Genetic Counseling"
    elif prob >= 0.65:
        acmg_class = "CLASS 3: LIKELY PATHOGENIC"
        symbol = "[*]"
        action = "Early Specialized Surveillance (Annual Contrast MRI / High-Definition Colonoscopy)"
    elif prob >= 0.35:
        acmg_class = "CLASS 2: VUS (UNCERTAIN)"
        symbol = "[?]"
        action = "Active Surveillance; DO NOT perform irreversible prophylactic surgeries"
    elif prob >= 0.10:
        acmg_class = "CLASS 1: LIKELY BENIGN"
        symbol = "[OK]"
        action = "Routine population lifestyle and age-based wellness screening"
    else:
        acmg_class = "CLASS 0: BENIGN"
        symbol = "[OK]"
        action = "Confirmed Harmless Polymorphism; Standard general population care"
        
    return {
        'gene': gene,
        'syndrome': syndrome,
        'logit': logit,
        'prob': prob,
        'acmg_class': acmg_class,
        'symbol': symbol,
        'action': action,
        'grantham': grantham,
        'delta_mw': delta_mw,
        'delta_charge': delta_charge
    }

def print_variant_report(patient_id, variant, res):
    print("\n" + "="*80)
    print("      GENOMEDL V2: VARIANT PATHOGENICITY CLINICAL REPORT")
    print(f"      Patient ID: {patient_id} | Reference Build: GRCh38 | Status: Calibrated")
    print("="*80)
    
    print("\n[1] PATHOGENICITY VERDICT:")
    print(f"    * Result: {res['symbol']} {res['acmg_class']}")
    print(f"    * Calibrated Probability: {res['prob']*100:.2f}% (Logit: {res['logit']:+.2f})")
    print(f"    * Clinical Action: {res['action']}")
    
    print("\n[2] MOLECULAR & BIOPHYSICAL EVIDENCE:")
    print(f"    * Gene Symbol:     {res['gene']} (Hereditary Group: {res['syndrome']})")
    print(f"    * Genomic Locus:   chr{variant.get('chrom', '.')}:{variant.get('pos', '.')} ({variant['ref']} > {variant['alt']})")
    print(f"    * Protein Change:  p.{variant.get('ref_aa', '.')}{variant.get('aa_pos', '.')}{variant.get('alt_aa', '.')}")
    print(f"    * Grantham Score:  {res['grantham']:.0f} (Canonical 20x20 distance matrix)")
    print(f"    * Molecular Deltas: ΔMW = {res['delta_mw']:+.1f} Da | ΔCharge = {res['delta_charge']:+.1f}")
    print("="*80 + "\n")

def main():
    parser = argparse.ArgumentParser(description="GenomeDL v2 Single-Variant Pathogenicity Predictor")
    parser.add_argument("--demo", action="store_true", help="Run Sarah's BRCA1 pathogenic case vs BRCA2 benign control")
    parser.add_argument("--gene", type=str, default="BRCA1", help="Target gene symbol")
    parser.add_argument("--ref", type=str, default="A", help="Reference nucleotide base")
    parser.add_argument("--alt", type=str, default="C", help="Mutated alternate base")
    parser.add_argument("--ref_aa", type=str, default="C", help="Original reference amino acid")
    parser.add_argument("--alt_aa", type=str, default="G", help="Mutated alternate amino acid")
    parser.add_argument("--aa_pos", type=int, default=61, help="Amino acid position along protein")
    parser.add_argument("--chrom", type=str, default="17", help="Chromosome identifier")
    parser.add_argument("--pos", type=int, default=43106487, help="Genomic coordinate (GRCh38)")
    parser.add_argument("--patient_id", type=str, default="QUERY_SAMPLE", help="Patient / Query ID")
    args = parser.parse_args()
    
    model, gene_to_id, syn_to_id, scaler, temp, device = load_v2_model()
    pairwise_lookup = build_pairwise_biophysical_lookup()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    prot_lens = get_protein_lengths_dict(os.path.join(script_dir, "out"))
    
    if args.demo:
        # Case 1: Sarah's Pathogenic Founder Mutation (BRCA1 p.Cys61Gly)
        v_sarah = {'gene': 'BRCA1', 'chrom': '17', 'pos': 43106487, 'ref': 'A', 'alt': 'C', 'ref_aa': 'C', 'alt_aa': 'G', 'aa_pos': 61, 'variant_class': 'missense'}
        res_sarah = predict_variant_pathogenicity(model, gene_to_id, syn_to_id, scaler, temp, device, v_sarah, pairwise_lookup, prot_lens)
        print_variant_report("DEMO_PATIENT_SARAH_24Y", v_sarah, res_sarah)
        
        # Case 2: Healthy Control Polymorphism (BRCA2 p.Asn372His)
        v_control = {'gene': 'BRCA2', 'chrom': '13', 'pos': 32332592, 'ref': 'A', 'alt': 'C', 'ref_aa': 'N', 'alt_aa': 'H', 'aa_pos': 372, 'variant_class': 'missense'}
        res_control = predict_variant_pathogenicity(model, gene_to_id, syn_to_id, scaler, temp, device, v_control, pairwise_lookup, prot_lens)
        print_variant_report("DEMO_PATIENT_BENCHMARK_CONTROL", v_control, res_control)
    else:
        var = {
            'gene': args.gene.upper(), 'chrom': str(args.chrom).replace('chr', ''),
            'pos': args.pos, 'ref': args.ref.upper(), 'alt': args.alt.upper(),
            'ref_aa': args.ref_aa.upper(), 'alt_aa': args.alt_aa.upper(),
            'aa_pos': args.aa_pos, 'variant_class': 'missense'
        }
        res = predict_variant_pathogenicity(model, gene_to_id, syn_to_id, scaler, temp, device, var, pairwise_lookup, prot_lens)
        print_variant_report(args.patient_id, var, res)

if __name__ == '__main__':
    main()
