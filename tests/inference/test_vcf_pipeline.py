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

import pytest

from genomedl.inference.vcf_pipeline import (
    DISCLAIMER_TEXT,
    VCFPipeline,
    VariantRecord,
)


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
        pos=45332803,
        id="rs34612342",
        ref="T",
        alt="C",
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


def test_spliceai_deep_intronic_safeguard(pipeline):
    """Novel deep-intronic variant with SpliceAI >= 0.50 must be classified as Likely Pathogenic."""
    deep_intronic = VariantRecord(
        chrom="17",
        pos=43080000,
        id=".",
        ref="G",
        alt="A",
        qual=100.0,
        filter_status="PASS",
        gene="BRCA1",
        consequence="intron_variant",
        spliceai_max=0.82,
    )
    verdict = pipeline.evaluate_variant(deep_intronic)
    assert verdict.tier == "Likely Pathogenic"
    assert "SpliceAI" in verdict.evidence_source
    assert verdict.calibrated_score >= 0.90


def test_spliceai_synonymous_splice_disruptor(pipeline):
    """Novel synonymous mutation with moderate SpliceAI delta score must be flagged as VUS."""
    synonymous_splice = VariantRecord(
        chrom="13",
        pos=32340000,
        id=".",
        ref="C",
        alt="T",
        qual=100.0,
        filter_status="PASS",
        gene="BRCA2",
        consequence="synonymous_variant",
        spliceai_max=0.35,
    )
    verdict = pipeline.evaluate_variant(synonymous_splice)
    assert verdict.tier == "VUS"
    assert "SpliceAI" in verdict.evidence_source


def test_missing_quality_or_genotype_rejected(pipeline, tmp_path):
    """Records missing GT, DP, or GQ (or having 0/0 / ./.) must not silently pass default filters."""
    vcf_content = """##fileformat=VCFv4.2
##reference=GRCh38
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
17\t43106487\trs1\tA\tC\t100\tPASS\tCSQ=C|missense_variant|MODERATE|BRCA1|||ENST00000357654
17\t43106487\trs2\tA\tC\t100\tPASS\tCSQ=C|missense_variant|MODERATE|BRCA1|||ENST00000357654\tGT\t0/1
17\t43106487\trs3\tA\tC\t100\tPASS\tCSQ=C|missense_variant|MODERATE|BRCA1|||ENST00000357654\tGT:DP:GQ\t./.:50:99
17\t43106487\trs4\tA\tC\t100\tPASS\tCSQ=C|missense_variant|MODERATE|BRCA1|||ENST00000357654\tGT:DP:GQ\t0/0:50:99
17\t43106487\trs5\tA\tC\t100\tPASS\tCSQ=C|missense_variant|MODERATE|BRCA1|||ENST00000357654\tGT:DP:GQ\t0/1:.:99
"""
    vcf_file = tmp_path / "missing_fields.vcf"
    vcf_file.write_text(vcf_content, encoding="utf-8")

    variants = pipeline.parse_vcf(vcf_file)
    assert variants == []


def test_filter_dot_rejected_by_default(pipeline, tmp_path):
    """FILTER='.' must be rejected when filter_pass_only=True and accepted only when filter_pass_only=False."""
    vcf_content = """##fileformat=VCFv4.2
##reference=GRCh38
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
17\t43106487\trs1\tA\tC\t100\t.\tCSQ=C|missense_variant|MODERATE|BRCA1|||ENST00000357654\tGT:DP:GQ\t0/1:50:99
"""
    vcf_file = tmp_path / "dot_filter.vcf"
    vcf_file.write_text(vcf_content, encoding="utf-8")

    assert pipeline.parse_vcf(vcf_file, filter_pass_only=True) == []
    unfiltered = pipeline.parse_vcf(vcf_file, filter_pass_only=False)
    assert len(unfiltered) == 1
    assert unfiltered[0].af is None


def test_unknown_reference_assembly_rejected(pipeline, tmp_path):
    """A VCF without positive GRCh38/hg38 header identification must raise ValueError."""
    vcf_content = """##fileformat=VCFv4.2
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
17\t43106487\trs1\tA\tC\t100\tPASS\tCSQ=C|missense_variant|MODERATE|BRCA1\tGT:DP:GQ\t0/1:50:99
"""
    vcf_file = tmp_path / "unknown_ref.vcf"
    vcf_file.write_text(vcf_content, encoding="utf-8")

    with pytest.raises(ValueError, match="GRCh38 / hg38"):
        pipeline.parse_vcf(vcf_file)


def test_multiallelic_gt_hom_and_spliceai_matching(pipeline, tmp_path):
    """Multiallelic rows must emit only called ALTs, bind homozygosity and SpliceAI per ALT, and preserve missing AF."""
    vcf_content = """##fileformat=VCFv4.2
##reference=GRCh38
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
17\t43080000\t.\tG\tA,T\t100\tPASS\tCSQ=A|intron_variant|MODIFIER|BRCA1,T|intron_variant|MODIFIER|BRCA1;SpliceAI=A|BRCA1|0.01|0.01|0.01|0.01,T|BRCA1|0.88|0.00|0.00|0.00;AF=.,0.0002\tGT:DP:GQ\t2/2:40:99
17\t43080010\t.\tC\tG,T\t100\tPASS\tCSQ=G|intron_variant|MODIFIER|BRCA1,T|intron_variant|MODIFIER|BRCA1;SpliceAI=G|BRCA1|0.02|0.00|0.00|0.00,T|BRCA1|0.91|0.00|0.00|0.00\tGT:DP:GQ\t1/2:45:99
"""
    vcf_file = tmp_path / "multiallelic.vcf"
    vcf_file.write_text(vcf_content, encoding="utf-8")

    variants = pipeline.parse_vcf(vcf_file)
    assert len(variants) == 3

    # First row (GT=2/2): only T (allele 2) is emitted, hom=True, SpliceAI=0.88, AF=0.0002
    assert variants[0].alt == "T"
    assert variants[0].is_biallelic is True
    assert variants[0].spliceai_max == pytest.approx(0.88)
    assert variants[0].af == pytest.approx(0.0002)

    # Second row (GT=1/2): both G and T emitted with is_biallelic=False and distinct SpliceAI scores; missing AF stays None
    g_var, t_var = variants[1], variants[2]
    assert g_var.alt == "G" and not g_var.is_biallelic
    assert g_var.spliceai_max == pytest.approx(0.02)
    assert g_var.af is None
    assert t_var.alt == "T" and not t_var.is_biallelic
    assert t_var.spliceai_max == pytest.approx(0.91)


def test_clinvar_conflicting_significance_maps_to_vus(pipeline):
    """Conflicting or mixed pathogenic+benign ClinVar labels must resolve to VUS, never Benign."""
    assert pipeline._classify_clinvar_significance("Conflicting classifications of pathogenicity")[0] == "VUS"
    assert pipeline._classify_clinvar_significance("Pathogenic; Benign")[0] == "VUS"
    assert pipeline._classify_clinvar_significance("Uncertain significance")[0] == "VUS"
    assert pipeline._classify_clinvar_significance("Pathogenic")[0] == "Pathogenic"


def test_overlapping_coordinate_fallback_is_not_guessed(pipeline, tmp_path):
    """An unannotated variant inside overlapping NTHL1/TSC2 intervals must not blindly pick the first gene."""
    vcf_content = """##fileformat=VCFv4.2
##reference=GRCh38
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tSAMPLE
chr16\t2047000\t.\tA\tG\t100\tPASS\t.\tGT:DP:GQ\t0/1:40:99
"""
    vcf_file = tmp_path / "overlap_nthl1_tsc2.vcf"
    vcf_file.write_text(vcf_content, encoding="utf-8")

    variants = pipeline.parse_vcf(vcf_file)
    assert variants == []


def test_compound_het_resolution_trans_cis_and_unphased(pipeline):
    """Multiple distinct heterozygous pathogenic calls in a recessive gene must resolve trans/unphased vs cis."""
    v1_trans = VariantRecord(
        chrom="1", pos=45332803, id="v1", ref="T", alt="C", qual=100.0, filter_status="PASS",
        gene="MUTYH", consequence="missense_variant", genotype="0|1", dp=40, gq=99,
        is_biallelic=False, phase_set="100",
    )
    v2_trans = VariantRecord(
        chrom="1", pos=45331833, id="v2", ref="C", alt="T", qual=100.0, filter_status="PASS",
        gene="MUTYH", consequence="frameshift_variant", genotype="1|0", dp=40, gq=99,
        is_biallelic=False, phase_set="100",
    )
    verdicts_trans = pipeline.evaluate_variants([v1_trans, v2_trans])
    assert all(v.tier in ("Pathogenic", "Likely Pathogenic") for v in verdicts_trans)
    assert all(len(v.cancer_risks) > 0 for v in verdicts_trans)
    assert "trans" in verdicts_trans[0].clinical_notes.lower()

    # Cis configuration (same haplotype slot in same PS block) stays Carrier (Recessive)
    v2_cis = VariantRecord(
        chrom="1", pos=45331833, id="v2", ref="C", alt="T", qual=100.0, filter_status="PASS",
        gene="MUTYH", consequence="frameshift_variant", genotype="0|1", dp=40, gq=99,
        is_biallelic=False, phase_set="100",
    )
    verdicts_cis = pipeline.evaluate_variants([v1_trans, v2_cis])
    assert all(v.tier == "Carrier (Recessive)" for v in verdicts_cis)
    assert all(v.cancer_risks == [] for v in verdicts_cis)
    assert "in cis" in verdicts_cis[0].clinical_notes.lower()

    # Unphased double het must NOT be treated as confirmed biallelic until trans phase is established
    v1_unphased = VariantRecord(
        chrom="1", pos=45332803, id="v1u", ref="T", alt="C", qual=100.0, filter_status="PASS",
        gene="MUTYH", consequence="missense_variant", genotype="0/1", dp=40, gq=99, is_biallelic=False,
    )
    v2_unphased = VariantRecord(
        chrom="1", pos=45331833, id="v2u", ref="C", alt="T", qual=100.0, filter_status="PASS",
        gene="MUTYH", consequence="frameshift_variant", genotype="0/1", dp=40, gq=99, is_biallelic=False,
    )
    verdicts_unphased = pipeline.evaluate_variants([v1_unphased, v2_unphased])
    assert all(v.tier == "Carrier (Recessive)" for v in verdicts_unphased)
    assert all(v.cancer_risks == [] for v in verdicts_unphased)
    assert "unphased" in verdicts_unphased[0].clinical_notes.lower()
    assert "phasing" in verdicts_unphased[0].clinical_notes.lower()


def test_risk_range_aggregation_preserves_per_gene_attribution(pipeline, tmp_path):
    """Summary risk entries must preserve per-gene range, horizon, and gene attribution without conflation."""
    from genomedl.inference.vcf_pipeline import VariantVerdict

    v_a = VariantVerdict(
        variant=VariantRecord(chrom="17", pos=1, id="a", ref="A", alt="C", qual=100.0, filter_status="PASS", gene="BRCA1"),
        tier="Pathogenic",
        evidence_source="test",
        cancer_risks=[{"outcome": "Breast cancer", "risk_low_pct": 40.0, "risk_high_pct": 75.0, "up_to_age": 80, "source": "StudyA", "evidence_level": "high"}],
    )
    v_b = VariantVerdict(
        variant=VariantRecord(chrom="13", pos=2, id="b", ref="G", alt="T", qual=100.0, filter_status="PASS", gene="BRCA2"),
        tier="Pathogenic",
        evidence_source="test",
        cancer_risks=[{"outcome": "Breast cancer", "risk_low_pct": 55.0, "risk_high_pct": 65.0, "up_to_age": 70, "source": "StudyB", "evidence_level": "high"}],
    )
    out_json = tmp_path / "risk_summary.json"
    pipeline.generate_report([v_a, v_b], output_json=out_json)

    import json
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    summaries = payload["cancer_risk_summary"]
    assert len(summaries) == 2
    by_gene = {tuple(s["genes"]): s for s in summaries}
    assert by_gene[("BRCA1",)]["lifetime_risk_range"] == "40% - 75%"
    assert by_gene[("BRCA1",)]["up_to_age"] == 80
    assert by_gene[("BRCA2",)]["lifetime_risk_range"] == "55% - 65%"
    assert by_gene[("BRCA2",)]["up_to_age"] == 70
