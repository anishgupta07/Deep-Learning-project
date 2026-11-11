"""High-level inference service helpers for the GenomeDL v3 frontend and API callers."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from genomedl.inference.vcf_pipeline import VCFPipeline
from genomedl.paths import FUSION_META_JSON, PANEL_JSON
from genomedl.reporting.pdf_report import generate_pdf_report

STANDARD_TIERS: List[str] = [
    "Pathogenic",
    "Likely Pathogenic",
    "Carrier (Recessive)",
    "VUS",
    "Likely Benign",
    "Benign",
]


def load_model_metadata(meta_path: Path = FUSION_META_JSON) -> Dict[str, Any]:
    """Load trained model metadata JSON, returning an empty dict if unavailable."""
    if not meta_path.exists():
        return {}
    with open(meta_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def load_panel_table(panel_path: Path = PANEL_JSON) -> pd.DataFrame:
    """Load the 130-gene reference panel JSON as a sorted DataFrame."""
    with open(panel_path, "r", encoding="utf-8") as handle:
        panel = json.load(handle)
    rows = [
        {
            "Gene": gene,
            "Syndrome Group": info.get("syndrome_group", "").replace("_", " "),
            "Strand": info.get("strand", ""),
            "Protein Length (aa)": int(info.get("protein_length", 0)),
            "MANE Transcript": info.get("mane_transcript", ""),
        }
        for gene, info in sorted(panel.items())
    ]
    return pd.DataFrame(rows)


def compute_tier_counts(verdicts: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count variant verdicts across standard clinical tiers."""
    counts = {tier: 0 for tier in STANDARD_TIERS}
    for verdict in verdicts:
        tier = str(verdict.get("tier", ""))
        counts[tier] = counts.get(tier, 0) + 1
    return counts


def run_vcf_analysis(
    pipeline: VCFPipeline,
    vcf_path: Path,
    sample_id: str = "DEMO-SAMPLE-001",
    min_dp: int = 10,
    min_gq: int = 20,
    filter_pass_only: bool = True,
) -> Dict[str, Any]:
    """Execute end-to-end VCF evaluation and render JSON/TXT/PDF in an isolated temp workspace."""
    variants = pipeline.parse_vcf(
        vcf_path,
        min_dp=min_dp,
        min_gq=min_gq,
        filter_pass_only=filter_pass_only,
    )
    verdicts = pipeline.evaluate_variants(variants)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        json_path = tmp_path / "vcf_report.json"
        pdf_path = tmp_path / "vcf_report.pdf"

        report_text = pipeline.generate_report(verdicts, output_json=json_path)
        report_json = json.loads(json_path.read_text(encoding="utf-8"))
        generate_pdf_report(json_path, pdf_path, sample_id=sample_id or "SAMPLE-001")
        pdf_bytes = pdf_path.read_bytes()

    verdict_dicts = report_json.get("verdicts", [])
    return {
        "sample_id": sample_id or "SAMPLE-001",
        "report_text": report_text,
        "report_json": report_json,
        "pdf_bytes": pdf_bytes,
        "tier_counts": compute_tier_counts(verdict_dicts),
    }


def analyze_uploaded_bytes(
    pipeline: VCFPipeline,
    file_bytes: bytes,
    filename: str,
    sample_id: str,
    min_dp: int = 10,
    min_gq: int = 20,
    filter_pass_only: bool = True,
) -> Dict[str, Any]:
    """Persist uploaded VCF bytes cleanly while preserving .vcf/.vcf.gz suffix and run analysis."""
    lower_name = filename.lower()
    if lower_name.endswith(".vcf.gz"):
        suffix = ".vcf.gz"
    elif lower_name.endswith(".vcf"):
        suffix = ".vcf"
    else:
        raise ValueError("Unsupported file format. Please upload a .vcf or .vcf.gz file.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_vcf = Path(tmp_dir) / f"uploaded{suffix}"
        tmp_vcf.write_bytes(file_bytes)
        return run_vcf_analysis(
            pipeline=pipeline,
            vcf_path=tmp_vcf,
            sample_id=sample_id,
            min_dp=min_dp,
            min_gq=min_gq,
            filter_pass_only=filter_pass_only,
        )
