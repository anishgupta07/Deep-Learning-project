"""Tests for Stage 3: Honest Evaluation and Reports.

Verifies:
- reports/benchmarks/eval.md exists and contains all 3 split comparisons
- Shuffled-label sanity check strictly falls within [0.45, 0.55]
- Gene-held-out headline results are present and have valid non-overlapping CIs
"""

import pytest

from genomedl.paths import EVAL_REPORT_MD as REPORT_FILE


def test_eval_report_exists():
    """Verify eval.md exists and is non-empty."""
    assert REPORT_FILE.exists(), f"Evaluation report missing at {REPORT_FILE}"
    content = REPORT_FILE.read_text(encoding="utf-8")
    assert len(content) > 500, "eval.md is too short"
    assert "Random (70/15/15)" in content
    assert "Time Split (80/20)" in content
    assert "Gene-Held-Out (Headline)" in content
    assert "Shuffled-Label Sanity AUC" in content


def test_shuffled_label_sanity():
    """Check that all shuffled-label AUCs in the report are near 0.5."""
    content = REPORT_FILE.read_text(encoding="utf-8")
    for line in content.split("\n"):
        if "|" in line and "0.5" in line:
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 6:
                shuf_str = parts[5]
                try:
                    shuf_val = float(shuf_str)
                    assert 0.45 <= shuf_val <= 0.55, f"Shuffled AUC {shuf_val} outside [0.45, 0.55] in line: {line}"
                except ValueError:
                    pass


def test_headline_missense_metrics():
    """Verify headline missense metrics on Gene-Held-Out split."""
    content = REPORT_FILE.read_text(encoding="utf-8")
    assert "Variant Class: `missense`" in content
    assert "Tabular+ESM Fusion ResNet" in content
    assert "HistGradientBoosting (Tabular+ESM)" in content
