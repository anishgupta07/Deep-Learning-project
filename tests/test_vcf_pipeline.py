"""Tests for Stage 4: End-to-End VCF Inference Pipeline.

Covers:
- Synthetic VCF with BRCA1 17:43106487 A>C and TP53 17:7675088 C>T (both pathogenic via ClinVar lookup)
- Unknown gene symbol raises clear ValueError
- hg19 / GRCh37 VCF reference rejection with liftover instructions
- Common variant filtering (gnomAD AF > 1% classified as Benign)
- Recessive zygosity logic for MUTYH / NTHL1 (heterozygous = carrier)
- Rule-based loss-of-function for frameshifts / canonical splice
- Mandatory disclaimer presence in clinical report
"""

import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v3.vcf_pipeline import (
    DISCLAIMER_TEXT,
    VCFPipeline,
    VariantRecord,
)

DATA_DIR = PROJECT_ROOT / "data"


@pytest.fixture(scope="module")
def pipeline():
    return VCFPipeline()


def test_synthetic_vcf_pathogenic_lookup(pipeline, tmp_path):
    """Synthetic VCF with BRCA1 and TP53 known pathogenic variants must be reported as Pathogenic via ClinVar."""
    vcf_content = """##fileformat=VCFv4.2
##reference=GRCh38
##INFO=<ID=CSQ,Number=.,Type=String,Description="Consequence annotations from Ensembl VEP">
##INFO=<ID=AF,Number=A,Type=Float,Description="Allele Frequency">
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth">
##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype Quality">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
17\t43106487\trs28897672\tA\tC\t100\tPASS\tCSQ=C|missense_variant|MODERATE|BRCA1|||ENST00000357654||||c.181T>G|p.Cys61Gly|||61|C/G||||;AF=0.00001\tGT:DP:GQ\t0/1:50:99
17\t7675088\trs28934578\tC\tT\t100\tPASS\tCSQ=T|missense_variant|MODERATE|TP53|||ENST00000269305||||c.524G>A|p.Arg175His|||175|R/H||||;AF=0.00002\tGT:DP:GQ\t0/1:60:99
"""
    vcf_file = tmp_path / "synthetic_test.vcf"
    vcf_file.write_text(vcf_content, encoding="utf-8")

    variants = pipeline.parse_vcf(vcf_file)
    assert len(variants) == 2, f"Expected 2 variants, got {len(variants)}"

    verdicts = [pipeline.evaluate_variant(v) for v in variants]
    assert len(verdicts) == 2

    # Check BRCA1 verdict
    brca1_v = verdicts[0]
    assert brca1_v.variant.gene == "BRCA1"
    assert brca1_v.tier == "Pathogenic"
    assert "ClinVar lookup" in brca1_v.evidence_source
    assert len(brca1_v.cancer_risks) > 0
    risk_outcomes = [r["outcome"] for r in brca1_v.cancer_risks]
    assert "Breast cancer" in risk_outcomes

    # Check TP53 verdict
    tp53_v = verdicts[1]
    assert tp53_v.variant.gene == "TP53"
    assert tp53_v.tier == "Pathogenic"
    assert "ClinVar lookup" in tp53_v.evidence_source
    assert len(tp53_v.cancer_risks) > 0

    # Check Report text and disclaimer
    report_text = pipeline.generate_report(verdicts)
    assert DISCLAIMER_TEXT in report_text
    assert "BRCA1" in report_text
    assert "TP53" in report_text


def test_unknown_gene_raises_error(pipeline):
    """A made-up gene symbol must immediately raise a ValueError."""
    fake_variant = VariantRecord(
        chrom="1",
        pos=100000,
        id=".",
        ref="A",
        alt="G",
        qual=100.0,
        filter_status="PASS",
        gene="NONEXISTENT_CANCER_GENE_999",
    )
    with pytest.raises(ValueError, match="not in 130-gene panel"):
        pipeline.evaluate_variant(fake_variant)


def test_hg19_rejection(pipeline, tmp_path):
    """A VCF using hg19 reference must be rejected with liftover instructions."""
    hg19_vcf_content = """##fileformat=VCFv4.2
##reference=hg19
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO
17\t41244435\t.\tA\tC\t100\tPASS\t.
"""
    vcf_file = tmp_path / "hg19_sample.vcf"
    vcf_file.write_text(hg19_vcf_content, encoding="utf-8")

    with pytest.raises(ValueError, match="hg19 / GRCh37.*liftOver"):
        pipeline.parse_vcf(vcf_file)


def test_common_variant_filtering(pipeline):
    """Variants with gnomAD AF > 1% (0.01) must be classified as Benign common polymorphism."""
    common_var = VariantRecord(
        chrom="17",
        pos=43100000,
        id="rs12345",
        ref="G",
        alt="A",
        qual=100.0,
        filter_status="PASS",
        gene="BRCA1",
        af=0.035,  # 3.5% AF
    )
    verdict = pipeline.evaluate_variant(common_var)
    assert verdict.tier == "Benign"
    assert "Common polymorphism" in verdict.evidence_source


def test_recessive_zygosity_carrier(pipeline):
    """Single heterozygous allele in recessive gene MUTYH must be reported as Carrier, not affected."""
    mutyh_var = VariantRecord(
        chrom="1",
        pos=45798000,
        id="rs34612342",
        ref="C",
        alt="T",
        qual=100.0,
        filter_status="PASS",
        gene="MUTYH",
        consequence="missense_variant",
        genotype="0/1",
        is_biallelic=False,
    )
    verdict = pipeline.evaluate_variant(mutyh_var)
    # Even if classified pathogenic, recessive single allele is Carrier
    assert "Carrier" in verdict.tier
    assert "recessive" in verdict.clinical_notes.lower()


def test_rule_based_frameshift(pipeline):
    """Frameshift variant in panel gene must be classified via Rule-based loss-of-function."""
    frameshift_var = VariantRecord(
        chrom="17",
        pos=43050000,
        id=".",
        ref="CT",
        alt="C",
        qual=100.0,
        filter_status="PASS",
        gene="BRCA1",
        consequence="frameshift_variant",
        af=0.00001,
    )
    verdict = pipeline.evaluate_variant(frameshift_var)
    assert verdict.tier == "Likely Pathogenic"
    assert "Rule-based" in verdict.evidence_source
