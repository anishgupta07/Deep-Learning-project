"""GenomeDL v3: Production Web Service & Interactive Clinical Inference Engine.

Integrates:
- FastAPI REST API (/healthz, /v1/predict/variant, /v1/predict/vcf, /docs)
- Interactive Gradio Clinical Web Interface (mounted at root '/')
- Full End-to-End Clinical VCF & Single Variant Decision Funnel
- Official 2-Page Hospital-Ready PDF Report Generation
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Project imports
from src.v3.generate_pdf_report import generate_pdf_report
from src.v3.vcf_pipeline import (
    DISCLAIMER_TEXT,
    RECESSIVE_GENES,
    VCFPipeline,
    VariantRecord,
    VariantVerdict,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("genomedl_service")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEMO_VCF_PATH = DATA_DIR / "sample_demo.vcf"

# Global pipeline instance (lazy/singleton loaded)
_pipeline: Optional[VCFPipeline] = None


def get_pipeline() -> VCFPipeline:
    global _pipeline
    if _pipeline is None:
        logger.info("Initializing GenomeDL v3 Multimodal Pipeline and warming weights...")
        _pipeline = VCFPipeline()
        logger.info("GenomeDL v3 Pipeline successfully loaded.")
    return _pipeline


# ------------------------------------------------------------------------------
# FastAPI Backend
# ------------------------------------------------------------------------------
fastapi_app = FastAPI(
    title="GenomeDL v3: Hereditary Cancer Genomic Interpretation API",
    description="Multimodal deep learning and clinical decision-support engine across 130 cancer predisposition genes.",
    version="3.0.0",
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SingleVariantRequest(BaseModel):
    chrom: str = Field(..., description="Chromosome without 'chr' prefix, e.g. '17'")
    pos: int = Field(..., description="1-based genomic coordinate (GRCh38), e.g. 43106487")
    ref: str = Field(..., description="Reference nucleotide allele, e.g. 'A'")
    alt: str = Field(..., description="Alternate nucleotide allele, e.g. 'C'")
    gene: str = Field(..., description="HGNC Gene symbol (must be in 130-gene panel), e.g. 'BRCA1'")
    consequence: str = Field(default="missense_variant", description="VEP consequence term")
    ref_aa: str = Field(default="", description="1-letter reference amino acid, e.g. 'C'")
    alt_aa: str = Field(default="", description="1-letter alternate amino acid, e.g. 'G'")
    aa_pos: int = Field(default=-1, description="1-based protein codon position, e.g. 61")
    af: float = Field(default=0.00001, description="gnomAD population allele frequency")
    genotype: str = Field(default="0/1", description="Zygosity string ('0/1' for het, '1/1' for hom)")
    dp: int = Field(default=50, description="Read depth")
    gq: int = Field(default=99, description="Genotype quality")


class RiskEstimate(BaseModel):
    outcome: str
    risk_range: str
    up_to_age: int
    source: str
    evidence_level: str


class SingleVariantResponse(BaseModel):
    gene: str
    chrom: str
    pos: int
    ref: str
    alt: str
    tier: str
    calibrated_score: Optional[float]
    evidence_source: str
    clinical_notes: str
    actionable_cancer_risks: List[RiskEstimate]
    disclaimer: str


@fastapi_app.get("/healthz", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "service": "GenomeDL v3 Clinical Interpretation API",
        "model_loaded": _pipeline is not None,
    }


@fastapi_app.post("/v1/predict/variant", response_model=SingleVariantResponse)
def api_predict_variant(req: SingleVariantRequest):
    pipeline = get_pipeline()
    var = VariantRecord(
        chrom=req.chrom.replace("chr", ""),
        pos=req.pos,
        id=".",
        ref=req.ref.upper(),
        alt=req.alt.upper(),
        qual=100.0,
        filter_status="PASS",
        gene=req.gene.upper(),
        consequence=req.consequence,
        ref_aa=req.ref_aa.upper(),
        alt_aa=req.alt_aa.upper(),
        aa_pos=req.aa_pos,
        af=req.af,
        genotype=req.genotype,
        dp=req.dp,
        gq=req.gq,
        is_biallelic=(req.genotype in ["1/1", "1|1"]),
    )

    try:
        verdict: VariantVerdict = pipeline.evaluate_variant(var)
        risks = [
            RiskEstimate(
                outcome=r["outcome"],
                risk_range=f"{r['risk_low_pct']:.0f}% - {r['risk_high_pct']:.0f}%",
                up_to_age=r["up_to_age"],
                source=r["source"],
                evidence_level=r.get("evidence_level", "high"),
            )
            for r in verdict.cancer_risks
        ]
        return SingleVariantResponse(
            gene=verdict.variant.gene,
            chrom=verdict.variant.chrom,
            pos=verdict.variant.pos,
            ref=verdict.variant.ref,
            alt=verdict.variant.alt,
            tier=verdict.tier,
            calibrated_score=round(verdict.calibrated_score, 4) if verdict.calibrated_score is not None else None,
            evidence_source=verdict.evidence_source,
            clinical_notes=verdict.clinical_notes,
            actionable_cancer_risks=risks,
            disclaimer=DISCLAIMER_TEXT,
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))


@fastapi_app.post("/v1/predict/vcf")
async def api_predict_vcf(file: UploadFile = File(...), min_dp: int = 10, min_gq: int = 20):
    if not (file.filename.endswith(".vcf") or file.filename.endswith(".vcf.gz")):
        raise HTTPException(status_code=400, detail="Only standard .vcf or .vcf.gz genomic files are supported.")

    pipeline = get_pipeline()
    suffix = ".vcf.gz" if file.filename.endswith(".vcf.gz") else ".vcf"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
        shutil.copyfileobj(file.file, tmp_file)
        tmp_path = Path(tmp_file.name)

    try:
        variants = pipeline.parse_vcf(tmp_path, min_dp=min_dp, min_gq=min_gq)
        verdicts = [pipeline.evaluate_variant(v) for v in variants]
        
        tmp_json = tmp_path.with_suffix(".json")
        pipeline.generate_report(verdicts, output_json=tmp_json)
        
        with open(tmp_json, "r", encoding="utf-8") as f:
            report_data = json.load(f)
            
        return report_data
    except Exception as e:
        logger.error(f"VCF parsing failed: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        if tmp_path.with_suffix(".json").exists():
            tmp_path.with_suffix(".json").unlink(missing_ok=True)


# ------------------------------------------------------------------------------
# Gradio UI Frontend
# ------------------------------------------------------------------------------
def get_available_panel_genes() -> List[str]:
    pipeline = get_pipeline()
    return sorted(list(pipeline.panel.keys()))


def ui_run_vcf_analysis(file_obj, min_dp: int, min_gq: int):
    """Processes uploaded VCF file or fallback demo, generates summary & PDF."""
    if file_obj is None:
        return "Please upload a .vcf file or click 'Load Demo VCF'.", None, None, None

    vcf_path = Path(file_obj.name)
    pipeline = get_pipeline()

    try:
        variants = pipeline.parse_vcf(vcf_path, min_dp=min_dp, min_gq=min_gq)
        if not variants:
            return "No variants passed the panel and quality filters (PASS, DP, GQ).", None, None, None

        verdicts = [pipeline.evaluate_variant(v) for v in variants]

        # Generate structured JSON and PDF report in temp dir
        temp_dir = Path(tempfile.mkdtemp(prefix="genomedl_"))
        out_json = temp_dir / "report.json"
        out_pdf = temp_dir / f"{vcf_path.stem}_Clinical_Report.pdf"

        report_txt = pipeline.generate_report(verdicts, output_json=out_json)
        pdf_path = generate_pdf_report(out_json, out_pdf, sample_id=vcf_path.stem.upper())

        # Build table for interactive UI display
        table_rows = []
        actionable_count = 0
        for v in verdicts:
            var = v.variant
            is_actionable = v.tier in ["Pathogenic", "Likely Pathogenic"]
            if is_actionable:
                actionable_count += 1

            risk_str = "; ".join([f"{r['outcome']}: {r['risk_low_pct']:.0f}-{r['risk_high_pct']:.0f}%" for r in v.cancer_risks]) or "N/A"
            table_rows.append({
                "Gene": var.gene,
                "Variant": f"chr{var.chrom}:{var.pos} {var.ref}>{var.alt}",
                "Consequence": var.hgvs_p or var.consequence,
                "Zygosity": var.genotype,
                "ACMG Tier": v.tier,
                "Evidence Source": v.evidence_source,
                "P(Pathogenic)": f"{v.calibrated_score:.4f}" if v.calibrated_score is not None else "-",
                "Lifetime Cancer Risks": risk_str,
            })

        df_summary = pd.DataFrame(table_rows)

        summary_metrics = f"""
### 📊 Analysis Summary for Sample: `{vcf_path.name}`
* **Total Panel Variants Evaluated:** **{len(verdicts)}**
* **Actionable Pathogenic / Likely Pathogenic:** <span style="color:red; font-weight:bold;">{actionable_count}</span>
* **Benign / Likely Benign / VUS:** **{len(verdicts) - actionable_count}**
"""
        return summary_metrics, df_summary, report_txt, str(pdf_path)

    except Exception as e:
        logger.error(f"UI VCF processing failed: {e}", exc_info=True)
        return f"❌ Error processing VCF: {str(e)}", None, None, None


def ui_load_demo():
    if DEMO_VCF_PATH.exists():
        return str(DEMO_VCF_PATH)
    return None


def ui_run_single_variant(
    gene: str,
    chrom: str,
    pos: int,
    ref: str,
    alt: str,
    consequence: str,
    ref_aa: str,
    alt_aa: str,
    aa_pos: int,
    af: float,
    genotype: str,
):
    """Processes single variant query."""
    pipeline = get_pipeline()
    var = VariantRecord(
        chrom=str(chrom).replace("chr", "").strip(),
        pos=int(pos),
        id=".",
        ref=ref.strip().upper(),
        alt=alt.strip().upper(),
        qual=100.0,
        filter_status="PASS",
        gene=gene.strip().upper(),
        consequence=consequence,
        ref_aa=ref_aa.strip().upper(),
        alt_aa=alt_aa.strip().upper(),
        aa_pos=int(aa_pos),
        af=float(af),
        genotype=genotype,
        dp=50,
        gq=99,
        is_biallelic=(genotype in ["1/1", "1|1"]),
    )

    try:
        verdict = pipeline.evaluate_variant(var)
        score_txt = f"{verdict.calibrated_score:.4f}" if verdict.calibrated_score is not None else "N/A"

        color_map = {
            "Pathogenic": "#dc2626",
            "Likely Pathogenic": "#ea580c",
            "VUS": "#d97706",
            "Likely Benign": "#16a34a",
            "Benign": "#059669",
            "Carrier (Recessive)": "#2563eb",
        }
        color = color_map.get(verdict.tier, "#475569")

        verdict_badge = f"""
<div style="background-color: {color}; color: white; padding: 12px 20px; border-radius: 8px; font-size: 1.25rem; font-weight: bold; text-align: center;">
  ACMG CLASSIFICATION: {verdict.tier.upper()}
</div>
"""
        risks_md = ""
        if verdict.cancer_risks:
            risks_md = "#### 🎗️ Published Lifetime Cancer Penetrance Risks:\n"
            for r in verdict.cancer_risks:
                risks_md += f"- **{r['outcome']}:** {r['risk_low_pct']:.0f}% - {r['risk_high_pct']:.0f}% (up to age {r['up_to_age']}) | *Source: {r['source']}*\n"
        else:
            risks_md = "*No direct high-penetrance lifetime cancer risk established for this variant status.*"

        details_md = f"""
### 🧬 Variant Details: `{gene}` chr{chrom}:{pos} {ref}>{alt}
* **Calibrated Model P(Pathogenic):** **`{score_txt}`**
* **Evidence Source:** `{verdict.evidence_source}`
* **Clinical Notes:** {verdict.clinical_notes}

{risks_md}

> ⚠️ **Mandatory Disclaimer:** {DISCLAIMER_TEXT}
"""
        return verdict_badge, details_md

    except Exception as e:
        return f"❌ Error: {str(e)}", ""


# Build Gradio Blocks Application
all_genes = get_available_panel_genes()

with gr.Blocks(title="GenomeDL v3: Hereditary Cancer AI Engine", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
# 🧬 GenomeDL v3: Multimodal Hereditary Cancer Interpretation Engine
### End-to-End Clinical Genomic Variant Interpretation across 130 Predisposition Genes
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![ESM-2](https://img.shields.io/badge/ESM--2-35M%20UR50D-green)](https://github.com/facebookresearch/esm)
[![RDKit](https://img.shields.io/badge/RDKit-Biophysics-blue)](https://www.rdkit.org/)
[![Test AUC](https://img.shields.io/badge/Test%20AUC-0.9929-success)](#)

---
"""
    )

    with gr.Tabs():
        # TAB 1: Patient VCF File Analysis
        with gr.Tab("📁 Whole-Exome / Panel VCF Analysis"):
            gr.Markdown(
                """
Upload a standard patient **VCF v4.2 file** (GRCh38 reference). The engine will parse variants, apply population frequency and ClinVar consensus filters, run **FusionResNet** on unclassified mutations, estimate organ-specific cancer penetrance, and generate a downloadable clinical PDF report.
"""
            )
            with gr.Row():
                with gr.Column(scale=1):
                    vcf_input = gr.File(label="Upload Patient VCF (.vcf / .vcf.gz)", file_types=[".vcf", ".vcf.gz"])
                    with gr.Row():
                        dp_slider = gr.Slider(minimum=5, maximum=100, value=10, step=5, label="Min Read Depth (DP)")
                        gq_slider = gr.Slider(minimum=10, maximum=99, value=20, step=5, label="Min Genotype Quality (GQ)")

                    with gr.Row():
                        btn_demo = gr.Button("📋 Load Demo VCF", variant="secondary")
                        btn_analyze = gr.Button("🚀 Run Clinical Pipeline", variant="primary")

                with gr.Column(scale=2):
                    summary_output = gr.Markdown("Upload a VCF to view clinical classification.")
                    pdf_download = gr.File(label="📄 Download 2-Page Clinical PDF Report")

            table_output = gr.Dataframe(label="Genomic Variant Interpretation Table", wrap=True)
            with gr.Accordion("📜 View Raw Clinical Text Report", open=False):
                txt_output = gr.Textbox(lines=15, label="Clinical Report Output")

            btn_demo.click(fn=ui_load_demo, inputs=[], outputs=[vcf_input])
            btn_analyze.click(
                fn=ui_run_vcf_analysis,
                inputs=[vcf_input, dp_slider, gq_slider],
                outputs=[summary_output, table_output, txt_output, pdf_download],
            )

        # TAB 2: Single Variant Predictor
        with gr.Tab("🔍 Single Variant Real-Time Calculator"):
            gr.Markdown("Directly evaluate any single nucleotide variant (SNV) against the 130 hereditary cancer gene panel.")
            with gr.Row():
                with gr.Column():
                    gene_dd = gr.Dropdown(choices=all_genes, value="BRCA1", label="Gene Symbol (130-Gene Panel)")
                    chrom_inp = gr.Textbox(value="17", label="Chromosome")
                    pos_inp = gr.Number(value=43106487, label="Genomic Coordinate (GRCh38)", precision=0)
                    with gr.Row():
                        ref_inp = gr.Textbox(value="A", label="Ref Allele")
                        alt_inp = gr.Textbox(value="C", label="Alt Allele")
                    csq_dd = gr.Dropdown(
                        choices=["missense_variant", "synonymous_variant", "stop_gained", "splice_region_variant", "intron_variant", "5_prime_UTR_variant", "3_prime_UTR_variant"],
                        value="missense_variant",
                        label="Consequence",
                    )
                    with gr.Row():
                        ref_aa_inp = gr.Textbox(value="C", label="Ref Amino Acid (1-letter)")
                        alt_aa_inp = gr.Textbox(value="G", label="Alt Amino Acid (1-letter)")
                        aa_pos_inp = gr.Number(value=61, label="Codon / AA Position", precision=0)
                    with gr.Row():
                        af_inp = gr.Number(value=0.00001, label="gnomAD Allele Frequency (AF)")
                        gt_dd = gr.Dropdown(choices=["0/1 (Heterozygous)", "1/1 (Homozygous)"], value="0/1 (Heterozygous)", label="Zygosity")

                    btn_eval = gr.Button("⚡ Evaluate Mutation", variant="primary")

                with gr.Column():
                    badge_output = gr.HTML()
                    details_output = gr.Markdown()

            btn_eval.click(
                fn=lambda g, c, p, r, a, cq, ra, aa, ap, af, gt: ui_run_single_variant(
                    g, c, p, r, a, cq, ra, aa, ap, af, "1/1" if "1/1" in gt else "0/1"
                ),
                inputs=[gene_dd, chrom_inp, pos_inp, ref_inp, alt_inp, csq_dd, ref_aa_inp, alt_aa_inp, aa_pos_inp, af_inp, gt_dd],
                outputs=[badge_output, details_output],
            )

        # TAB 3: Developer API & Architecture Docs
        with gr.Tab("📚 REST API & Architecture"):
            gr.Markdown(
                """
### 🔌 Programmatic Access via REST API
This Hugging Face Space runs a live **FastAPI** service in addition to this Gradio interface:
* **Interactive Swagger UI:** Visit [`/docs`](/docs) on this Space's base URL.
* **Health Check:** `GET /healthz`
* **Single Variant Prediction:** `POST /v1/predict/variant`
* **Batch VCF Analysis:** `POST /v1/predict/vcf`

#### Example Python Request:
```python
import requests

url = "https://<your-space-name>.hf.space/v1/predict/variant"
payload = {
    "chrom": "17",
    "pos": 43106487,
    "ref": "A",
    "alt": "C",
    "gene": "BRCA1",
    "consequence": "missense_variant",
    "ref_aa": "C",
    "alt_aa": "G",
    "aa_pos": 61,
    "af": 0.00001
}
response = requests.post(url, json=payload)
print(response.json())
```

---
### 🧬 Multimodal Fusion Architecture
* **Tabular Branch:** Grantham chemical distance, molecular weight delta ($\Delta$MW), charge delta, topological polar surface area ($\Delta$TPSA), and Tanimoto chemical fingerprint distances.
* **ESM-2 Language Model Branch:** 480-dimensional embeddings from Facebook AI's 35M-parameter `esm2_t12_35M_UR50D` + masked marginal log-likelihood ratio scores.
* **Calibration:** Temperature-scaled posterior probability with validation Expected Calibration Error (ECE) = **0.0009**.
"""
            )

    gr.Markdown(
        f"""
---
<small style="color: gray;">
<b>Regulatory Notice:</b> {DISCLAIMER_TEXT} | Model version: 3.0.0
</small>
"""
    )

# Mount Gradio into FastAPI
app = gr.mount_gradio_app(fastapi_app, demo, path="/")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
