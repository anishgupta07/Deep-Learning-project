"""Tests for Clinical Genomics PDF Report Generation."""

from pathlib import Path
import pytest

pypdf = pytest.importorskip("pypdf")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
from src.v3.generate_pdf_report import generate_pdf_report


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

    p2_text = reader.pages[1].extract_text()
    assert "Detailed Genomic Variant Classifications" in p2_text
    assert "MANDATORY CLINICAL GENOMICS DISCLAIMER" in p2_text
