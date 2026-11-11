"""Tests for the Streamlit frontend and inference service helpers."""

from __future__ import annotations

import gzip
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from genomedl.inference.service import (
    analyze_uploaded_bytes,
    load_model_metadata,
    load_panel_table,
    run_vcf_analysis,
)
from genomedl.inference.vcf_pipeline import VCFPipeline
from genomedl.paths import DEMO_VCF, PROJECT_ROOT


@pytest.fixture(scope="module")
def pipeline() -> VCFPipeline:
    """Load the inference pipeline once per test module."""
    return VCFPipeline()


def test_load_panel_table() -> None:
    """Reference panel loader should return all 130 canonical genes with expected columns."""
    df = load_panel_table()
    assert len(df) == 130
    assert list(df.columns) == [
        "Gene",
        "Syndrome Group",
        "Strand",
        "Protein Length (aa)",
        "MANE Transcript",
    ]
    assert "BRCA1" in set(df["Gene"])


def test_load_model_metadata(tmp_path: Path) -> None:
    """Checkpoint metadata should expose AUC metrics and 5-seed ensemble count."""
    meta = load_model_metadata()
    assert "test_auc" in meta
    assert meta.get("num_seeds") == 5
    assert load_model_metadata(tmp_path / "missing_meta.json") == {}


def test_run_vcf_analysis_demo(pipeline: VCFPipeline) -> None:
    """End-to-end service analysis on the demo VCF should match expected verdicts and PDF bytes."""
    result = run_vcf_analysis(
        pipeline=pipeline,
        vcf_path=DEMO_VCF,
        sample_id="DEMO-TEST-001",
    )
    report_json = result["report_json"]
    assert result["sample_id"] == "DEMO-TEST-001"
    assert report_json["total_variants"] == 5
    assert report_json["actionable_count"] == 2
    assert result["tier_counts"]["Carrier (Recessive)"] == 1
    assert result["pdf_bytes"].startswith(b"%PDF")
    assert len(result["pdf_bytes"]) > 1000
    assert "GENOMEDL v3: CLINICAL GENOMICS INFERENCE REPORT" in result["report_text"]


def test_analyze_uploaded_bytes_plain_and_gzip(pipeline: VCFPipeline) -> None:
    """Uploaded raw .vcf and gzipped .vcf.gz payloads should analyze identically."""
    raw_bytes = DEMO_VCF.read_bytes()

    res_plain = analyze_uploaded_bytes(
        pipeline=pipeline,
        file_bytes=raw_bytes,
        filename="uploaded_demo.vcf",
        sample_id="UPLOAD-PLAIN",
    )
    res_gzip = analyze_uploaded_bytes(
        pipeline=pipeline,
        file_bytes=gzip.compress(raw_bytes),
        filename="uploaded_demo.vcf.gz",
        sample_id="UPLOAD-GZIP",
    )

    for res in (res_plain, res_gzip):
        assert res["report_json"]["total_variants"] == 5
        assert res["report_json"]["actionable_count"] == 2
        assert res["tier_counts"]["Carrier (Recessive)"] == 1
        assert res["pdf_bytes"].startswith(b"%PDF")


def test_analyze_uploaded_bytes_rejects_invalid_extension(pipeline: VCFPipeline) -> None:
    """Unsupported upload extensions should raise a clear ValueError."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        analyze_uploaded_bytes(
            pipeline=pipeline,
            file_bytes=b"not a vcf",
            filename="notes.txt",
            sample_id="INVALID",
        )


def test_streamlit_app_smoke() -> None:
    """Streamlit AppTest should execute app.py cleanly and render auto-run demo results."""
    at = AppTest.from_file(str(PROJECT_ROOT / "app.py"), default_timeout=120)
    at.run()

    assert not at.exception
    assert len(at.title) == 1
    assert "GenomeDL v3" in at.title[0].value

    metric_map = {m.label: m.value for m in at.metric}
    assert metric_map["Panel Variants"] == "5"
    assert metric_map["Actionable (P / LP)"] == "2"
    assert metric_map["Recessive Carriers"] == "1"
    assert len(at.tabs) == 4
