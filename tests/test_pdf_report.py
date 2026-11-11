"""Tests for Clinical Genomics PDF Report Generation."""

from datetime import datetime, timezone
import json
from pathlib import Path
import pytest

pypdf = pytest.importorskip("pypdf")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from src.v3.generate_pdf_report import classify_genotype_zygosity, generate_pdf_report


def test_pdf_report_generation(tmp_path):
    json_path = PROJECT_ROOT / "reports" / "vcf_report.json"
    assert json_path.exists(), "vcf_report.json must exist before testing PDF generation"

    out_pdf = tmp_path / "test_report.pdf"
    pdf_path = generate_pdf_report(json_path, out_pdf, sample_id="TEST-SAMPLE")

    assert pdf_path.exists()
    assert pdf_path.stat().st_size > 5000, "PDF file must not be empty"

    reader = pypdf.PdfReader(str(pdf_path))
    assert len(reader.pages) == 2, f"Expected 2-page compact layout, got {len(reader.pages)}"

    p1_text = reader.pages[0].extract_text()
    assert "GENOMEDL CLINICAL GENOMICS PIPELINE" in p1_text
    assert "Executive Summary: Lifetime Cancer Risk" in p1_text
    assert "65% - 72%" in p1_text
    assert "50% - 60%" in p1_text
    utc_date = datetime.now(timezone.utc).strftime("%B %d, %Y")
    assert utc_date in p1_text

    p2_text = reader.pages[1].extract_text()
    assert "Detailed Genomic Variant Classifications" in p2_text
    assert "MANDATORY CLINICAL GENOMICS DISCLAIMER" in p2_text
    assert "Platt logit scaling" in p2_text
    assert "0.9919" in p2_text
    assert "T=0.9357" not in p2_text


def test_pdf_genotype_zygosity_and_unknown_baseline(tmp_path):
    """Verify genotype zygosity formatting and missing baseline handling without 5.0% invention."""
    assert classify_genotype_zygosity("0/0") == "Hom-Ref"
    assert classify_genotype_zygosity("2/2") == "Hom"
    assert classify_genotype_zygosity("2|2") == "Hom"
    assert classify_genotype_zygosity("./.") == "No-call"
    assert classify_genotype_zygosity("0/1") == "Het"

    sample_json = {
        "total_variants": 2,
        "actionable_count": 1,
        "cancer_risk_summary": [
            {
                "cancer_type": "Rare Ependymoma (Unknown Pop Baseline)",
                "lifetime_risk_range": "20% - 30%",
                "genes": ["NF2"],
                "up_to_age": 70,
            }
        ],
        "verdicts": [
            {
                "gene": "NF2", "chrom": "22", "pos": 29603998, "id": "rs1", "ref": "A", "alt": "G,T",
                "hgvs_c": "", "hgvs_p": "p.Met1Val", "consequence": "missense_variant",
                "genotype": "2/2", "dp": 42, "gq": 99, "af": 0.0001,
                "tier": "Pathogenic", "evidence_source": "ClinVar", "calibrated_score": 0.999,
                "cancer_risks": [], "other_syndrome_features": [], "notes": "",
            },
            {
                "gene": "BRCA1", "chrom": "17", "pos": 43100000, "id": "rs2", "ref": "G", "alt": "A",
                "hgvs_c": "", "hgvs_p": "", "consequence": "synonymous",
                "genotype": "0/0", "dp": 35, "gq": 99, "af": 0.0,
                "tier": "Benign", "evidence_source": "Model", "calibrated_score": 0.001,
                "cancer_risks": [], "other_syndrome_features": [], "notes": "",
            },
        ],
    }
    in_json = tmp_path / "edge_report.json"
    in_json.write_text(json.dumps(sample_json), encoding="utf-8")
    out_pdf = tmp_path / "edge_report.pdf"

    generate_pdf_report(in_json, out_pdf, sample_id="EDGE-TEST")
    reader = pypdf.PdfReader(str(out_pdf))
    full_text = " ".join(" ".join(page.extract_text() for page in reader.pages).split())
    assert "Baseline unavailable" in full_text
    assert "5.0% general pop." not in full_text
    assert "2/2 (Hom, DP=42)" in full_text
    assert "0/0 (Hom-Ref, DP=35)" in full_text
