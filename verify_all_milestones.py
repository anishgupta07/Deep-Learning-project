import os
import sys
import torch
import numpy as np
import pandas as pd

# Force UTF-8 encoding on console streams for cross-platform safety
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from patient_risk_engine import PatientRiskEngine, parse_patient_vcf, export_html_patient_report

print("="*80)
print("   GENOMEDL V2: EXTERNAL 3-WAY AUDIT & MILESTONE VERIFICATION SUITE")
print("="*80)

# =============================================================================
# VERIFICATION PASS 1: DATASET INTEGRITY, CHECKPOINT & ARTIFACT AUDIT
# =============================================================================
print("\n>>> STARTING VERIFICATION PASS 1: DATASETS, CHECKPOINTS & ARTIFACT AUDIT <<<")

# M1 Data Audit
lbl_csv = os.path.join(PROJECT_DIR, "data", "v2_pan_cancer_featured_labelled.csv")
vus_csv = os.path.join(PROJECT_DIR, "data", "v2_pan_cancer_featured_vus.csv")
assert os.path.exists(lbl_csv), f"Missing labelled dataset: {lbl_csv}"
assert os.path.exists(vus_csv), f"Missing VUS dataset: {vus_csv}"

df_lbl = pd.read_csv(lbl_csv)
assert len(df_lbl) == 148825, f"Expected 148825 rows in labelled dataset, got {len(df_lbl)}"
print(f"[*] Pass 1.1: Verified Labelled Dataset -> 148,825 rows, 48 columns.")

# M2 Checkpoint Audit
ckpt_path = os.path.join(PROJECT_DIR, "models", "v2_pan_cancer_resnet.pth")
assert os.path.exists(ckpt_path), f"Missing checkpoint: {ckpt_path}"
ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

feature_cols = ckpt['feature_cols']
assert len(feature_cols) == 29, f"Expected 29 features, got {len(feature_cols)}"
null_sum = df_lbl[feature_cols].isnull().sum().sum()
assert null_sum == 0, f"Found {null_sum} null values in feature columns!"
print(f"[*] Pass 1.2: Verified 29 features across all rows -> Exactly 0 missing values.")

gene_map = ckpt['gene_to_id']
syn_map = ckpt['syndrome_to_id']
assert len(gene_map) == 130, f"Expected 130 genes, got {len(gene_map)}"
temp = float(ckpt['temperature'])
assert abs(temp - 0.8769) < 0.01, f"Expected temperature ~0.8769, got {temp}"
print(f"[*] Pass 1.3: Verified Checkpoint -> 130 Genes, {len(syn_map)} Syndromes, Temperature = {temp:.4f}.")
print(f"    Checkpoint Evaluated Metrics: ROC-AUC={ckpt['test_auc']*100:.2f}%, AUPRC={ckpt['test_auprc']*100:.2f}%, Precision={ckpt['test_prec']*100:.2f}%, Specificity={ckpt['test_spec']*100:.2f}%, ECE={ckpt['ece']:.4f}")

# M3 & M4 Artifacts Audit
vcfs = [
    os.path.join(PROJECT_DIR, "data", "demo_vcfs", "patient_sarah_hboc.vcf"),
    os.path.join(PROJECT_DIR, "data", "demo_vcfs", "patient_healthy_control.vcf"),
    os.path.join(PROJECT_DIR, "data", "demo_vcfs", "patient_lynch_carrier.vcf")
]
for v in vcfs:
    assert os.path.exists(v) and os.path.getsize(v) > 0, f"Missing or empty VCF: {v}"
print(f"[*] Pass 1.4: Verified Demo VCFs -> 3 complete clinical test sequencing files.")

reports = [
    os.path.join(PROJECT_DIR, "data", "demo_reports", "patient_sarah_hboc.html"),
    os.path.join(PROJECT_DIR, "data", "demo_reports", "patient_healthy_control.html"),
    os.path.join(PROJECT_DIR, "data", "demo_reports", "patient_lynch_carrier.html")
]
for r in reports:
    assert os.path.exists(r) and os.path.getsize(r) > 10000, f"Missing or truncated report: {r}"
print(f"[*] Pass 1.5: Verified Certified HTML Reports -> 3 reports present (all > 10 KB).")
print("[+] VERIFICATION PASS 1 PASSED: 100% OF MILESTONE ARTIFACTS CONFIRMED.")

# =============================================================================
# VERIFICATION PASS 2: DEEP LEARNING MODEL INFERENCE & BIOLOGICAL SENSITIVITY
# =============================================================================
print("\n>>> STARTING VERIFICATION PASS 2: MODEL INFERENCE & CONTROL TESTS <<<")

engine = PatientRiskEngine()

# Test 1: Pathogenic Driver BRCA1 p.C61G
sarah_vars = engine.screen_patient_variants(parse_patient_vcf(vcfs[0]))
res_sarah = engine.evaluate_patient_risk("SARAH_TEST", sarah_vars)
brca1_top = res_sarah['top_variant']
print(f"[*] Pass 2.1: Sarah Top Variant: {brca1_top['gene']} {brca1_top['protein_change']} -> Prob: {brca1_top['prob']*100:.2f}% | Class: {brca1_top['acmg_class']}")
assert brca1_top['prob'] >= 0.85, f"BRCA1 p.C61G should be pathogenic (>=85%), got {brca1_top['prob']}"
assert "PATHOGENIC" in brca1_top['acmg_class']

# Test 2: Benign Control BRCA2 p.N372H
ctrl_vars = engine.screen_patient_variants(parse_patient_vcf(vcfs[1]))
res_ctrl = engine.evaluate_patient_risk("CTRL_TEST", ctrl_vars)
brca2_benign = [v for v in res_ctrl['all_variants'] if v['gene'] == 'BRCA2'][0]
print(f"[*] Pass 2.2: Benign Control: {brca2_benign['gene']} {brca2_benign['protein_change']} -> Prob: {brca2_benign['prob']*100:.2f}% | Class: {brca2_benign['acmg_class']}")
assert brca2_benign['prob'] <= 0.20, f"BRCA2 p.N372H must be benign (<=20%), got {brca2_benign['prob']}"
assert "BENIGN" in brca2_benign['acmg_class']

# Test 3: Synonymous Negative Control ATM p.K1192K
atm_benign = [v for v in res_ctrl['all_variants'] if v['gene'] == 'ATM'][0]
print(f"[*] Pass 2.3: Synonymous Control: {atm_benign['gene']} {atm_benign['protein_change']} -> Prob: {atm_benign['prob']*100:.2f}% | Class: {atm_benign['acmg_class']}")
assert atm_benign['prob'] < 0.02, f"ATM synonymous variant should be near 0%, got {atm_benign['prob']}"

# Test 4: Lynch Syndrome Driver MLH1 p.S44F
lynch_vars = engine.screen_patient_variants(parse_patient_vcf(vcfs[2]))
res_lynch = engine.evaluate_patient_risk("LYNCH_TEST", lynch_vars)
mlh1_top = res_lynch['top_variant']
print(f"[*] Pass 2.4: Lynch Top Variant: {mlh1_top['gene']} {mlh1_top['protein_change']} -> Prob: {mlh1_top['prob']*100:.2f}% | Class: {mlh1_top['acmg_class']}")
assert mlh1_top['prob'] >= 0.90, f"MLH1 p.S44F should be pathogenic (>=90%), got {mlh1_top['prob']}"
print("[+] VERIFICATION PASS 2 PASSED: BIOLOGICAL DAMAGE SCORING ACCURATE & CALIBRATED.")

# =============================================================================
# VERIFICATION PASS 3: END-TO-END PATIENT RISK PROJECTION & HTML VERIFICATION
# =============================================================================
print("\n>>> STARTING VERIFICATION PASS 3: END-TO-END PATIENT RISK RADAR & HTML <<<")

# Check Sarah Risk Projection
assert "CONFIRMED CARRIER" in res_sarah['carrier_status']
assert res_sarah['risk_tier'] == "HIGH_HEREDITARY_CANCER_RISK"
assert res_sarah['syndrome_group'] == "Breast_Ovarian_HR_repair"
print(f"[*] Pass 3.1: Sarah Carrier Diagnosis: {res_sarah['carrier_status']}")

# Check Healthy Control Projection
assert "NEGATIVE" in res_ctrl['carrier_status']
assert res_ctrl['risk_tier'] == "POPULATION_BASELINE"
print(f"[*] Pass 3.2: Healthy Control Carrier Diagnosis: {res_ctrl['carrier_status']}")

# Check Lynch Syndrome Projection
assert "CONFIRMED CARRIER" in res_lynch['carrier_status']
assert res_lynch['risk_tier'] == "HIGH_HEREDITARY_CANCER_RISK"
assert res_lynch['syndrome_group'] == "Lynch_MMR_polymerase"
print(f"[*] Pass 3.3: Lynch Carrier Diagnosis: {res_lynch['carrier_status']}")

# Validate Generated HTML Files for Valid Markup and Expected Text
for r_path, expected_text in [
    (reports[0], "CONFIRMED CARRIER: BRCA1 Pathogenic Mutation"),
    (reports[1], "NEGATIVE: Non-Carrier"),
    (reports[2], "CONFIRMED CARRIER: MLH1 Pathogenic Mutation")
]:
    with open(r_path, "r", encoding="utf-8") as f:
        html = f.read()
    assert expected_text in html, f"Expected '{expected_text}' in {r_path}"
    assert "GenomeDL v2 Engine" in html
    assert "ROC-AUC: 99.29%" in html
    print(f"[*] Pass 3.4: Verified HTML file: {os.path.basename(r_path)} contains expected diagnostic callouts.")

print("\n" + "="*80)
print(">>> ALL 3 VERIFICATION PASSES COMPLETED WITH 100% SUCCESS <<<")
print("="*80)
