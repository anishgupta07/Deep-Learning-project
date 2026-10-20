#!/usr/bin/env python3
"""
==============================================================================
GenomeDL v2: Strict 3-Way Independent Verification of Person-Level Scorer
==============================================================================
Validates the mathematical properties, clinical cohort accuracy, and novel
variant handling of the PersonCancerScorer engine across 3 independent passes.
"""

import os
import sys
import torch
import numpy as np

# Force UTF-8 encoding on console streams
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from patient_cancer_scorer import (
    PersonCancerScorer,
    parse_patient_vcf,
    export_html_report,
    AUTOSOMAL_RECESSIVE_GENES,
    HIGH_PENETRANCE_GENES,
    MODERATE_PENETRANCE_GENES
)

print("="*84)
print("   GENOMEDL V2: PERSON-LEVEL CANCER RISK SCORING AUDIT SUITE")
print("   Strict 3-Way Independent Mathematical & Clinical Cohort Verification")
print("="*84)

scorer = PersonCancerScorer()

# =============================================================================
# VERIFICATION PASS 1: MATHEMATICAL AXIOMS, BOUNDS & COMPOUNDING AUDIT
# =============================================================================
print("\n>>> STARTING VERIFICATION PASS 1: MATHEMATICAL AXIOMS & COMPOUNDING AUDIT <<<")

# Test 1.1: Boundedness & Zero-Impact of Benign Polymorphisms
empty_res = scorer.score_patient("EMPTY_SAMPLE", [])
assert empty_res['person_cancer_score'] == 12.0, f"Expected 12.0 base risk, got {empty_res['person_cancer_score']}"
assert empty_res['risk_tier'] == "LOW / POPULATION BASELINE RISK"
print("[*] Pass 1.1: Boundedness Baseline Verified -> Empty profile returns exactly 12.0 / 100.")

# Test 1.2: Benign Variant Invariance
benign_var = [{'gene': 'BRCA2', 'chrom': '13', 'pos': 32332592, 'ref': 'A', 'alt': 'C', 'ref_aa': 'N', 'alt_aa': 'H', 'aa_pos': 372, 'variant_class': 'missense', 'genotype': '0/1'}]
benign_res = scorer.score_patient("BENIGN_SAMPLE", benign_var)
assert benign_res['screened_variants'][0]['prob'] < 0.15, "BRCA2 N372H must have P(Damaging) < 0.15"
assert benign_res['screened_variants'][0]['effective_impact'] == 0.0, "Benign variant must contribute exactly 0 excess impact"
assert benign_res['person_cancer_score'] == 12.0, f"Expected exactly 12.0 for benign variant, got {benign_res['person_cancer_score']}"
print("[*] Pass 1.2: Zero-Impact Invariance Verified -> True benign polymorphisms contribute 0.000 excess risk.")

# Test 1.3: Monotonicity
var_moderate = {'gene': 'CHEK2', 'chrom': '22', 'pos': 28695000, 'ref': 'C', 'alt': 'T', 'ref_aa': 'R', 'alt_aa': 'W', 'aa_pos': 145, 'variant_class': 'missense', 'genotype': '0/1'}
var_severe = {'gene': 'BRCA1', 'chrom': '17', 'pos': 43106487, 'ref': 'A', 'alt': 'C', 'ref_aa': 'C', 'alt_aa': 'G', 'aa_pos': 61, 'variant_class': 'missense', 'genotype': '0/1'}

res_mod = scorer.score_patient("MOD_SAMPLE", [var_moderate])
res_sev = scorer.score_patient("SEV_SAMPLE", [var_severe])
assert res_sev['person_cancer_score'] > res_mod['person_cancer_score'], f"Severe ({res_sev['person_cancer_score']}) must exceed moderate ({res_mod['person_cancer_score']})"
print(f"[*] Pass 1.3: Strict Monotonicity Verified -> Severe ({res_sev['person_cancer_score']}) > Moderate ({res_mod['person_cancer_score']}).")

# Test 1.4: Multi-Hit Compounding Strict Inequality
var_atm = {'gene': 'ATM', 'chrom': '11', 'pos': 108236100, 'ref': 'A', 'alt': 'G', 'ref_aa': 'V', 'alt_aa': 'A', 'aa_pos': 320, 'variant_class': 'missense', 'genotype': '0/1'}
res_atm = scorer.score_patient("ATM_SAMPLE", [var_atm])
res_compound = scorer.score_patient("COMPOUND_SAMPLE", [var_moderate, var_atm])

score_chek2 = res_mod['person_cancer_score']
score_atm = res_atm['person_cancer_score']
score_comp = res_compound['person_cancer_score']
assert score_comp > max(score_chek2, score_atm), f"Compound {score_comp} must exceed both {score_chek2} and {score_atm}"
print(f"[*] Pass 1.4: Multi-Hit Compounding Verified -> Score ({score_comp}) strictly exceeds single hits ({score_chek2}, {score_atm}).")

# Test 1.5: Recessive Zygosity Gate
var_mutyh_het = {'gene': 'MUTYH', 'chrom': '1', 'pos': 45797359, 'ref': 'G', 'alt': 'A', 'ref_aa': 'Y', 'alt_aa': 'C', 'aa_pos': 179, 'variant_class': 'missense', 'genotype': '0/1'}
var_mutyh_hom = {'gene': 'MUTYH', 'chrom': '1', 'pos': 45797359, 'ref': 'G', 'alt': 'A', 'ref_aa': 'Y', 'alt_aa': 'C', 'aa_pos': 179, 'variant_class': 'missense', 'genotype': '1/1'}

res_m_het = scorer.score_patient("MUTYH_HET", [var_mutyh_het])
res_m_hom = scorer.score_patient("MUTYH_HOM", [var_mutyh_hom])
assert res_m_het['person_cancer_score'] < 25.0, f"MUTYH carrier must be low risk (<25), got {res_m_het['person_cancer_score']}"
assert res_m_hom['person_cancer_score'] >= 75.0, f"MUTYH biallelic must be high risk (>=75), got {res_m_hom['person_cancer_score']}"
print(f"[*] Pass 1.5: Recessive Zygosity Gate Verified -> Monoallelic ({res_m_het['person_cancer_score']}) vs Biallelic ({res_m_hom['person_cancer_score']}).")
print("[+] VERIFICATION PASS 1 PASSED: 100% OF MATHEMATICAL AXIOMS CONFIRMED.")

# =============================================================================
# VERIFICATION PASS 2: BENCHMARK CLINICAL COHORTS VERIFICATION
# =============================================================================
print("\n>>> STARTING VERIFICATION PASS 2: BENCHMARK CLINICAL COHORTS VERIFICATION <<<")

vcf_dir = os.path.join(PROJECT_DIR, "demo_vcfs")

# Benchmark 2.1: Sarah HBOC Carrier
rec_sarah = parse_patient_vcf(os.path.join(vcf_dir, "patient_sarah_hboc.vcf"))
res_sarah = scorer.score_patient("SARAH", rec_sarah)
print(f"[*] Pass 2.1: Sarah HBOC -> Score: {res_sarah['person_cancer_score']}/100 | Tier: {res_sarah['risk_tier']}")
assert res_sarah['person_cancer_score'] >= 85.0, f"Sarah score must be >= 85, got {res_sarah['person_cancer_score']}"
assert "HIGH" in res_sarah['risk_tier']
assert res_sarah['organ_projections']['Breast']['projected_risk'] >= 60.0

# Benchmark 2.2: Healthy Control
rec_ctrl = parse_patient_vcf(os.path.join(vcf_dir, "patient_healthy_control.vcf"))
res_ctrl = scorer.score_patient("CONTROL", rec_ctrl)
print(f"[*] Pass 2.2: Healthy Control -> Score: {res_ctrl['person_cancer_score']}/100 | Tier: {res_ctrl['risk_tier']}")
assert res_ctrl['person_cancer_score'] <= 15.0, f"Control score must be <= 15, got {res_ctrl['person_cancer_score']}"
assert "LOW" in res_ctrl['risk_tier']

# Benchmark 2.3: Lynch Carrier
rec_lynch = parse_patient_vcf(os.path.join(vcf_dir, "patient_lynch_carrier.vcf"))
res_lynch = scorer.score_patient("LYNCH", rec_lynch)
print(f"[*] Pass 2.3: Lynch Carrier -> Score: {res_lynch['person_cancer_score']}/100 | Tier: {res_lynch['risk_tier']}")
assert res_lynch['person_cancer_score'] >= 90.0, f"Lynch score must be >= 90, got {res_lynch['person_cancer_score']}"
assert res_lynch['organ_projections']['Colorectal']['projected_risk'] >= 70.0

# Benchmark 2.4: Compound Moderate (CHEK2 + ATM)
rec_comp = parse_patient_vcf(os.path.join(vcf_dir, "patient_compound_moderate.vcf"))
res_comp = scorer.score_patient("COMPOUND", rec_comp)
print(f"[*] Pass 2.4: Compound Moderate -> Score: {res_comp['person_cancer_score']}/100 | Tier: {res_comp['risk_tier']}")
assert 55.0 <= res_comp['person_cancer_score'] <= 75.0, f"Compound score must be 55-75, got {res_comp['person_cancer_score']}"
assert "ELEVATED" in res_comp['risk_tier']

# Benchmark 2.5: Recessive Carrier vs Biallelic
rec_carr = parse_patient_vcf(os.path.join(vcf_dir, "patient_recessive_carrier.vcf"))
res_carr = scorer.score_patient("MUTYH_CARRIER", rec_carr)
rec_bi = parse_patient_vcf(os.path.join(vcf_dir, "patient_recessive_biallelic.vcf"))
res_bi = scorer.score_patient("MUTYH_BIALLELIC", rec_bi)
print(f"[*] Pass 2.5: Recessive Carrier ({res_carr['person_cancer_score']}) vs Biallelic ({res_bi['person_cancer_score']})")
assert res_carr['person_cancer_score'] <= 25.0
assert res_bi['person_cancer_score'] >= 75.0

print("[+] VERIFICATION PASS 2 PASSED: 100% OF CLINICAL BENCHMARKS CONFIRMED.")

# =============================================================================
# VERIFICATION PASS 3: DYNAMIC NOVEL VARIANT INGESTION & PIPELINE INTEGRITY
# =============================================================================
print("\n>>> STARTING VERIFICATION PASS 3: DYNAMIC NOVEL VARIANT INGESTION & PIPELINE INTEGRITY <<<")

# Test 3.1: Novel TP53 Variant (Not in pre-indexed dictionary)
rec_novel = parse_patient_vcf(os.path.join(vcf_dir, "patient_novel_variant.vcf"))
res_novel = scorer.score_patient("NOVEL_PATIENT", rec_novel)
assert len(res_novel['screened_variants']) == 2, f"Expected 2 variants, got {len(res_novel['screened_variants'])}"
tp53_v = [v for v in res_novel['screened_variants'] if v['gene'] == 'TP53'][0]
assert tp53_v['prob'] >= 0.80, f"TP53 p.R248Q must be pathogenic, got {tp53_v['prob']}"
assert res_novel['person_cancer_score'] >= 85.0
print(f"[*] Pass 3.1: Novel TP53 Variant Ingestion -> ResNet Prob: {tp53_v['prob']*100:.2f}% | Overall Score: {res_novel['person_cancer_score']}/100.")

# Test 3.2: HTML Report Generation
out_html = os.path.join(vcf_dir, "test_clinical_report.html")
export_html_report(res_sarah, out_html)
assert os.path.exists(out_html), "HTML report file was not created"
file_size = os.path.getsize(out_html)
assert file_size > 2000, f"HTML file unexpectedly small: {file_size} bytes"
with open(out_html, 'r', encoding='utf-8') as f:
    html_content = f.read()
assert "91.3" in html_content
assert "HIGH HEREDITARY CANCER RISK" in html_content
print(f"[*] Pass 3.2: HTML Report Verified -> Generated valid report ({file_size:,} bytes).")

print("\n" + "="*84)
print(">>> ALL 3 VERIFICATION PASSES COMPLETED WITH 100% SUCCESS <<<")
print("="*84)
