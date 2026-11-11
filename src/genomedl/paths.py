"""Canonical repository paths and configuration loader for GenomeDL v3."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIGS_DIR = PROJECT_ROOT / "configs"
DEFAULT_CONFIG = CONFIGS_DIR / "fusion_v3.yaml"

DATA_DIR = PROJECT_ROOT / "data"
DATA_REFERENCE_DIR = DATA_DIR / "reference"
DATA_PROCESSED_DIR = DATA_DIR / "processed"
DATA_CACHE_DIR = DATA_DIR / "cache"
DATA_DEMO_DIR = DATA_DIR / "demo"

PANEL_JSON = DATA_REFERENCE_DIR / "panel.json"
PANEL_COORDS_JSON = DATA_REFERENCE_DIR / "panel_coordinates.json"
GENE_STRAND_JSON = DATA_REFERENCE_DIR / "gene_strand.json"
PROTEIN_LENGTHS_JSON = DATA_REFERENCE_DIR / "protein_lengths.json"
PANEL_FASTA = DATA_REFERENCE_DIR / "panel_proteins.fasta"
PENETRANCE_CSV = DATA_REFERENCE_DIR / "penetrance.csv"

CLINVAR_LABELLED_CSV = DATA_PROCESSED_DIR / "clinvar_labelled_v3.csv"
FEATURED_SNV_CSV = DATA_PROCESSED_DIR / "v3_featured_snv.csv"

ESM_CACHE_PARQUET = DATA_CACHE_DIR / "esm_cache.parquet"
ESM_EMBEDDINGS_PT = DATA_CACHE_DIR / "esm_embeddings.pt"

DEMO_VCF = DATA_DEMO_DIR / "sample_demo.vcf"

MODELS_DIR = PROJECT_ROOT / "models"
FUSION_V3_DIR = MODELS_DIR / "fusion_v3"
FUSION_MODEL_PTH = FUSION_V3_DIR / "model.pth"
FUSION_META_JSON = FUSION_V3_DIR / "meta.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
REPORTS_BENCHMARKS_DIR = REPORTS_DIR / "benchmarks"
REPORTS_DEMO_DIR = REPORTS_DIR / "demo_vcf"
REPORTS_HG002_DIR = REPORTS_DIR / "giab_hg002"

EVAL_REPORT_MD = REPORTS_BENCHMARKS_DIR / "eval.md"
DEMO_REPORT_TXT = REPORTS_DEMO_DIR / "vcf_report.txt"
DEMO_REPORT_JSON = REPORTS_DEMO_DIR / "vcf_report.json"
DEMO_REPORT_PDF = REPORTS_DEMO_DIR / "vcf_report.pdf"
HG002_REPORT_TXT = REPORTS_HG002_DIR / "hg002_report.txt"
HG002_REPORT_JSON = REPORTS_HG002_DIR / "hg002_report.json"
HG002_REPORT_PDF = REPORTS_HG002_DIR / "hg002_report.pdf"


def resolve_project_path(path_value: str | Path) -> Path:
    """Resolve a relative path against PROJECT_ROOT while preserving absolute paths."""
    candidate = Path(path_value)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def load_config(config_path: Optional[Path] = None) -> Dict[str, Any]:
    """Load YAML hyperparameters and paths configuration, returning {} if absent."""
    target = config_path if config_path is not None else DEFAULT_CONFIG
    if not target.exists():
        return {}
    with open(target, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    return data if isinstance(data, dict) else {}
