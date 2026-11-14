"""GenomeDL v3 interactive clinical genomics Streamlit application."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import streamlit as st

# Ensure local src/ package is importable if run directly without editable install
_SRC_DIR = Path(__file__).resolve().parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from genomedl.inference.service import (  # noqa: E402
    STANDARD_TIERS,
    analyze_uploaded_bytes,
    load_model_metadata,
    load_panel_table,
    run_vcf_analysis,
)
from genomedl.inference.vcf_pipeline import DISCLAIMER_TEXT, VCFPipeline  # noqa: E402
from genomedl.paths import DEMO_VCF, EVAL_REPORT_MD  # noqa: E402
from genomedl.reporting.pdf_report import (  # noqa: E402
    POPULATION_BASELINES,
    classify_genotype_zygosity,
    create_risk_chart,
)

st.set_page_config(
    page_title="GenomeDL v3 — Hereditary Cancer VCF Interpreter",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource(show_spinner="Loading 130-gene panel, ClinVar index, ESM cache, and FusionResNet ensemble...")
def get_pipeline() -> VCFPipeline:
    """Initialize and cache the VCF inference pipeline once per server process."""
    return VCFPipeline()


@st.cache_data
def get_cached_panel() -> pd.DataFrame:
    """Load the 130-gene reference table once across reruns."""
    return load_panel_table()


@st.cache_data
def get_cached_metadata() -> Dict[str, Any]:
    """Load model checkpoint metadata once across reruns."""
    return load_model_metadata()


@st.cache_data
def get_eval_markdown() -> str:
    """Load out-of-gene evaluation markdown if present."""
    return EVAL_REPORT_MD.read_text(encoding="utf-8") if EVAL_REPORT_MD.exists() else ""


def _render_sidebar(meta: Dict[str, Any]) -> Dict[str, Any]:
    """Render input controls, QC filters, and model snapshot in the sidebar."""
    with st.sidebar:
        st.header("Analysis Inputs")
        input_mode = st.radio(
            "VCF Input Mode",
            options=["Demo VCF (sample_demo.vcf)", "Upload Custom VCF (.vcf / .vcf.gz)"],
            index=0,
        )

        uploaded_file = None
        default_sample_id = "DEMO-SAMPLE-001"
        if input_mode == "Upload Custom VCF (.vcf / .vcf.gz)":
            uploaded_file = st.file_uploader(
                "Choose a GRCh38 VCF file",
                type=["vcf", "gz"],
                help="Supports uncompressed .vcf or gzipped .vcf.gz aligned to GRCh38 / hg38.",
            )
            if uploaded_file is not None:
                stem = uploaded_file.name.replace(".vcf.gz", "").replace(".vcf", "")
                default_sample_id = stem.upper() or "UPLOADED-SAMPLE"

        sample_id = st.text_input("Sample Identifier", value=default_sample_id)

        st.subheader("Quality Control Filters")
        min_dp = st.slider("Minimum Read Depth (DP)", min_value=0, max_value=100, value=10, step=1)
        min_gq = st.slider("Minimum Genotype Quality (GQ)", min_value=0, max_value=99, value=20, step=1)
        filter_pass_only = st.checkbox("Require FILTER == PASS", value=True)

        run_clicked = st.button("Run Analysis", type="primary", use_container_width=True)

        if meta:
            st.divider()
            st.subheader("Model Snapshot")
            c1, c2 = st.columns(2)
            c1.metric("Test AUC", f"{meta.get('test_auc', 0.0):.4f}")
            c2.metric("Test AUPRC", f"{meta.get('test_auprc', 0.0):.4f}")
            st.caption(
                f"5-Seed FusionResNet | Platt ECE: {meta.get('ece_before', 0.0):.4f} → {meta.get('ece_after', 0.0):.4f} | "
                f"Gene Embedding: {'ON' if meta.get('use_gene_embedding') else 'OFF'}"
            )

    return {
        "input_mode": input_mode,
        "uploaded_file": uploaded_file,
        "sample_id": sample_id.strip() or "SAMPLE-001",
        "min_dp": min_dp,
        "min_gq": min_gq,
        "filter_pass_only": filter_pass_only,
        "run_clicked": run_clicked,
    }


def _render_overview_tab(result: Dict[str, Any]) -> None:
    """Render KPI metrics, actionable alert banner, per-gene risk table, and chart."""
    report_json = result["report_json"]
    counts = result["tier_counts"]
    total = int(report_json.get("total_variants", 0))
    actionable = int(report_json.get("actionable_count", 0))
    carriers = counts.get("Carrier (Recessive)", 0)
    vus = counts.get("VUS", 0)
    benign_total = counts.get("Benign", 0) + counts.get("Likely Benign", 0)

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Panel Variants", total)
    k2.metric("Actionable (P / LP)", actionable)
    k3.metric("Recessive Carriers", carriers)
    k4.metric("VUS", vus)
    k5.metric("Benign / Likely Benign", benign_total)

    if actionable > 0:
        st.error(
            f"**{actionable} actionable pathogenic/likely pathogenic variant(s) detected** "
            f"for sample `{result['sample_id']}`."
        )
    else:
        st.success(
            f"No high-penetrance pathogenic or likely pathogenic variants identified for sample `{result['sample_id']}`."
        )

    cancer_risks: List[Dict[str, Any]] = report_json.get("cancer_risk_summary", [])
    if cancer_risks:
        st.subheader("Executive Lifetime Cancer Risk Summary (Per-Gene Attribution)")
        risk_rows = []
        for item in cancer_risks:
            outcome = item["cancer_type"]
            pop_base = POPULATION_BASELINES.get(outcome)
            risk_rows.append(
                {
                    "Cancer Type / Organ": outcome,
                    "Estimated Lifetime Risk": item["lifetime_risk_range"],
                    "Gene": ", ".join(item.get("genes", [])),
                    "General Population Baseline": f"~{pop_base:.1f}%" if pop_base is not None else "Unavailable",
                }
            )
        st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)

        chart_buf = create_risk_chart(cancer_risks)
        if chart_buf is not None:
            st.subheader("Carrier Risk vs. General Population Baseline")
            st.image(chart_buf, use_container_width=True)
    else:
        st.info("No elevated lifetime penetrance estimates triggered for this sample.")


def _render_verdicts_tab(result: Dict[str, Any]) -> None:
    """Render filterable variant verdict table and detailed clinical evidence cards."""
    verdicts: List[Dict[str, Any]] = result["report_json"].get("verdicts", [])
    if not verdicts:
        st.info("No candidate variants in the 130-gene panel passed the active QC filters.")
        return

    present_tiers = [t for t in STANDARD_TIERS if any(v.get("tier") == t for v in verdicts)]
    f1, f2 = st.columns([2, 1])
    selected_tiers = f1.multiselect(
        "Filter by ACMG-Style Tier",
        options=present_tiers or STANDARD_TIERS,
        default=present_tiers or STANDARD_TIERS,
    )
    gene_query = f2.text_input("Filter by Gene Symbol", placeholder="e.g. BRCA1, TP53, MUTYH").strip().upper()

    filtered = [
        v
        for v in verdicts
        if v.get("tier") in selected_tiers and (not gene_query or gene_query in str(v.get("gene", "")).upper())
    ]

    rows = []
    for v in filtered:
        gt = v.get("genotype", ".")
        zyg = classify_genotype_zygosity(gt)
        dp = v.get("dp")
        gq = v.get("gq")
        af = v.get("af")
        score = v.get("calibrated_score")
        rows.append(
            {
                "Gene": v.get("gene", ""),
                "GRCh38 Locus": f"chr{v.get('chrom')}:{v.get('pos')} {v.get('ref')}>{v.get('alt')}",
                "HGVS / Consequence": v.get("hgvs_p") or v.get("hgvs_c") or v.get("consequence") or "—",
                "Genotype": f"{gt} ({zyg}, DP={dp}, GQ={gq})",
                "gnomAD AF": f"{af:.5f}" if af is not None else "Missing",
                "Tier": v.get("tier", ""),
                "Calibrated P": round(float(score), 4) if score is not None else None,
                "Evidence": v.get("evidence_source", ""),
            }
        )

    st.caption(f"Showing {len(filtered)} of {len(verdicts)} evaluated variants.")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.subheader("Detailed Variant Notes & Penetrance Citations")
    max_cards = 50
    for idx, v in enumerate(filtered[:max_cards], start=1):
        label = (
            f"#{idx} {v.get('gene')} (chr{v.get('chrom')}:{v.get('pos')} {v.get('ref')}>{v.get('alt')}) "
            f"— {v.get('tier')}"
        )
        expanded = v.get("tier") in ("Pathogenic", "Likely Pathogenic", "Carrier (Recessive)")
        with st.expander(label, expanded=expanded):
            st.markdown(f"**Evidence:** {v.get('evidence_source', '')}")
            if v.get("calibrated_score") is not None:
                st.markdown(f"**Calibrated P(pathogenic):** `{v['calibrated_score']:.4f}`")
            st.markdown(f"**Clinical Notes:** {v.get('notes', '')}")
            if v.get("cancer_risks"):
                st.markdown("**Published Lifetime Cancer Risks:**")
                for risk in v["cancer_risks"]:
                    st.markdown(
                        f"- {risk['outcome']}: **{risk['risk_low_pct']:.0f}%–{risk['risk_high_pct']:.0f}%** "
                        f"(Evidence: {risk['evidence_level']})"
                    )
            if v.get("other_syndrome_features"):
                st.markdown("**Other Syndromic Features (Non-Cancer):**")
                for feat in v["other_syndrome_features"]:
                    st.markdown(
                        f"- {feat['outcome']}: {feat['risk_low_pct']:.0f}%–{feat['risk_high_pct']:.0f}%"
                    )
    if len(filtered) > max_cards:
        st.caption(f"Showing detail cards for the first {max_cards} matching variants; download JSON/TXT for all records.")


def _render_reports_tab(result: Dict[str, Any]) -> None:
    """Render clinical report download buttons and raw TXT/JSON previews."""
    stem = result["sample_id"].lower().replace(" ", "_")
    d1, d2, d3 = st.columns(3)
    d1.download_button(
        "Download Clinical PDF Report",
        data=result["pdf_bytes"],
        file_name=f"{stem}_genomedl_report.pdf",
        mime="application/pdf",
        use_container_width=True,
    )
    d2.download_button(
        "Download Structured JSON",
        data=json.dumps(result["report_json"], indent=2),
        file_name=f"{stem}_genomedl_report.json",
        mime="application/json",
        use_container_width=True,
    )
    d3.download_button(
        "Download Text Report",
        data=result["report_text"],
        file_name=f"{stem}_genomedl_report.txt",
        mime="text/plain",
        use_container_width=True,
    )

    with st.expander("Text Report Preview", expanded=True):
        st.code(result["report_text"], language="text")

    with st.expander("Structured JSON Preview", expanded=False):
        st.json(result["report_json"])


def _render_panel_tab(panel_df: pd.DataFrame, eval_md: str) -> None:
    """Render searchable 130-gene reference panel and honest out-of-gene benchmarks."""
    st.subheader("130-Gene Hereditary Cancer Predisposition Panel")
    search = st.text_input("Search Gene or Syndrome Group", placeholder="e.g. BRCA, Lynch, Fanconi").strip().lower()
    shown_df = (
        panel_df[
            panel_df["Gene"].str.lower().str.contains(search)
            | panel_df["Syndrome Group"].str.lower().str.contains(search)
        ]
        if search
        else panel_df
    )
    st.caption(f"Displaying {len(shown_df)} of {len(panel_df)} canonical panel genes.")
    st.dataframe(shown_df, use_container_width=True, hide_index=True)

    if eval_md:
        st.divider()
        st.markdown(eval_md)


def main() -> None:
    """Render the GenomeDL v3 Streamlit application."""
    st.title("🧬 GenomeDL v3 — Clinical Genomics Variant Interpreter")
    st.caption(
        "130-Gene Germline Hereditary Cancer Panel • Multimodal ESM-2 Fusion ResNet (5-Seed Ensemble + Platt Calibration)"
    )
    st.warning(
        f"**Mandatory Disclaimer:** {DISCLAIMER_TEXT} "
        "Do not upload identifiable clinical patient VCFs to public deployments."
    )

    meta = get_cached_metadata()
    panel_df = get_cached_panel()
    eval_md = get_eval_markdown()
    controls = _render_sidebar(meta)

    pipeline = get_pipeline()

    # Auto-run the synthetic demo once on initial load so visitors immediately see an active report
    should_run = controls["run_clicked"]
    if "analysis_result" not in st.session_state and controls["input_mode"] == "Demo VCF (sample_demo.vcf)":
        should_run = True

    if should_run:
        try:
            with st.spinner("Running GRCh38 QC, ClinVar lookup, FusionResNet inference, and PDF report generation..."):
                if controls["input_mode"] == "Demo VCF (sample_demo.vcf)":
                    st.session_state["analysis_result"] = run_vcf_analysis(
                        pipeline=pipeline,
                        vcf_path=DEMO_VCF,
                        sample_id=controls["sample_id"],
                        min_dp=controls["min_dp"],
                        min_gq=controls["min_gq"],
                        filter_pass_only=controls["filter_pass_only"],
                    )
                elif controls["uploaded_file"] is not None:
                    uploaded = controls["uploaded_file"]
                    st.session_state["analysis_result"] = analyze_uploaded_bytes(
                        pipeline=pipeline,
                        file_bytes=uploaded.getvalue(),
                        filename=uploaded.name,
                        sample_id=controls["sample_id"],
                        min_dp=controls["min_dp"],
                        min_gq=controls["min_gq"],
                        filter_pass_only=controls["filter_pass_only"],
                    )
                else:
                    st.warning("Please upload a `.vcf` or `.vcf.gz` file before running custom analysis.")
        except (ValueError, OSError) as exc:
            st.error(f"VCF Analysis Error: {exc}")

    result = st.session_state.get("analysis_result")
    tab_overview, tab_verdicts, tab_reports, tab_panel = st.tabs(
        ["Overview & Cancer Risk", "Variant Verdicts", "Reports & Downloads", "130-Gene Panel & Benchmarks"]
    )

    with tab_overview:
        if result:
            _render_overview_tab(result)
        else:
            st.info("Select a VCF input mode in the sidebar and click **Run Analysis**.")

    with tab_verdicts:
        if result:
            _render_verdicts_tab(result)
        else:
            st.info("Run an analysis to explore variant verdicts.")

    with tab_reports:
        if result:
            _render_reports_tab(result)
        else:
            st.info("Run an analysis to preview and download clinical reports.")

    with tab_panel:
        _render_panel_tab(panel_df, eval_md)


if __name__ == "__main__":
    main()
