"""Stage 4: End-to-End VCF Clinical Genomics Inference Pipeline.

Pipeline Flow:
  1. Input Validation: Check GRCh38 reference header; halt with liftover instructions if hg19/b37.
  2. Parse & Filter: FILTER=PASS, min DP (>=10), min GQ (>=20), split multi-allelics.
  3. Panel Filter: Restrict to 130 hereditary cancer genes from data/panel.json.
  4. Common Variant Filter: gnomAD AF > 1% (0.01) -> Benign (common polymorphism).
  5. ClinVar Lookup: Match (chrom, pos, ref, alt); if stars >= 2, use ClinVar verdict (no model).
  6. Rule-based LoF: Frameshift, large deletion, or canonical splice -> "predicted loss-of-function".
  7. Fusion Model Inference: Unclassified SNVs evaluated with FusionResNet + ESM masked marginals.
  8. Zygosity Resolution: Dominant vs recessive (MUTYH, NTHL1 require biallelic for high risk).
  9. Clinical Risk Report: Published carrier risk ranges from data/penetrance.csv (no formula!).
     Non-cancer syndrome features listed separately.
     Mandatory clinical research disclaimer printed.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.v3.stage0.build_v3_features import (
    AA_PROPS,
    CLASS_TO_ID,
    GRANTHAM,
    VARIANT_CLASSES,
)
from src.v3.train_fusion import (
    BINARY_COLS,
    CONTINUOUS_COLS,
    FusionResNet,
    TemperatureScaler,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vcf_pipeline")

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_CHECKPOINT = MODELS_DIR / "genome_dl_fusion_v3.pth"
DEFAULT_PENETRANCE = DATA_DIR / "penetrance.csv"
DEFAULT_CLINVAR_CSV = DATA_DIR / "clinvar_labelled_v3.csv"
DEFAULT_PANEL_JSON = DATA_DIR / "panel.json"
DEFAULT_ESM_CACHE = DATA_DIR / "esm_cache.parquet"
DEFAULT_FASTA = DATA_DIR / "panel_proteins.fasta"

RECESSIVE_GENES = {"MUTYH", "NTHL1"}
DISCLAIMER_TEXT = (
    "Research prototype, not a diagnosis. Confirm in a clinical lab with genetic counselling. "
    "Testing children for adult-onset conditions needs counselling."
)


@dataclass
class VariantRecord:
    chrom: str
    pos: int
    id: str
    ref: str
    alt: str
    qual: float
    filter_status: str
    gene: str
    transcript: str = ""
    consequence: str = ""
    ref_aa: str = ""
    alt_aa: str = ""
    aa_pos: int = -1
    hgvs_c: str = ""
    hgvs_p: str = ""
    af: float = 0.0
    genotype: str = "0/1"
    dp: int = 30
    gq: int = 99
    is_biallelic: bool = False
    spliceai_max: float = 0.0


@dataclass
class VariantVerdict:
    variant: VariantRecord
    tier: str  # Pathogenic, Likely Pathogenic, VUS, Likely Benign, Benign, Carrier
    evidence_source: str  # ClinVar (stars >= 2), Rule-based LoF, Common Variant, Fusion Model
    calibrated_score: Optional[float] = None
    cancer_risks: List[Dict[str, Union[str, float]]] = field(default_factory=list)
    other_syndrome_features: List[Dict[str, Union[str, float]]] = field(default_factory=list)
    clinical_notes: str = ""


class VCFPipeline:
    """Production clinical genomics inference pipeline for pediatric and germline VCFs."""

    def __init__(
        self,
        checkpoint_path: Path = DEFAULT_CHECKPOINT,
        penetrance_csv: Path = DEFAULT_PENETRANCE,
        clinvar_csv: Path = DEFAULT_CLINVAR_CSV,
        panel_json_path: Path = DEFAULT_PANEL_JSON,
        esm_cache_path: Path = DEFAULT_ESM_CACHE,
        protein_fasta_path: Path = DEFAULT_FASTA,
        device: Optional[str] = None,
    ):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Initializing VCFPipeline on {self.device}...")

        # 1. Panel metadata
        self.panel = json.load(open(panel_json_path, "r", encoding="utf-8"))
        self.syndromes = sorted(list(set(info["syndrome_group"] for info in self.panel.values())))
        self.syndrome_to_id = {s: i for i, s in enumerate(self.syndromes)}
        logger.info(f"Loaded panel with {len(self.panel)} genes and {len(self.syndromes)} syndrome groups.")

        # 2. ClinVar Lookup Index
        self.clinvar_index = {}
        if clinvar_csv.exists():
            logger.info(f"Building fast ClinVar lookup index from {clinvar_csv}...")
            c_df = pd.read_csv(clinvar_csv, low_memory=False)
            for _, r in c_df.iterrows():
                key = (str(r["chrom"]).replace("chr", ""), int(r["pos"]), str(r["ref"]), str(r["alt"]))
                self.clinvar_index[key] = {
                    "gene": str(r["gene"]),
                    "clinical_significance": str(r["clinical_significance"]),
                    "stars": int(r["stars"]) if pd.notna(r["stars"]) else 0,
                    "review_status": str(r["review_status"]),
                    "variation_id": str(r["variation_id"]),
                    "variant_class": str(r["variant_class"]),
                }
            logger.info(f"Indexed {len(self.clinvar_index)} ClinVar variants.")

        # 3. Penetrance table
        self.penetrance_df = pd.read_csv(penetrance_csv) if penetrance_csv.exists() else pd.DataFrame()
        logger.info(f"Loaded {len(self.penetrance_df)} penetrance entries from {penetrance_csv}.")

        # 4. ESM Cache
        self.esm_cache = {}
        if esm_cache_path.exists():
            logger.info(f"Loading ESM masked-marginal cache from {esm_cache_path}...")
            ec_df = pd.read_parquet(esm_cache_path)
            for _, r in ec_df.iterrows():
                key = (str(r["gene"]), int(r["aa_pos"]))
                emb = np.array(r["embedding"], dtype=np.float32)
                log_probs = {c.replace("log_p_", ""): float(r[c]) for c in ec_df.columns if c.startswith("log_p_")}
                self.esm_cache[key] = (log_probs, emb)
            logger.info(f"Loaded {len(self.esm_cache)} cached ESM positions.")

        # 5. Protein sequences
        self.gene_seqs = {}
        if protein_fasta_path.exists():
            curr_g, curr_s = "", []
            with open(protein_fasta_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith(">"):
                        if curr_g:
                            self.gene_seqs[curr_g] = "".join(curr_s)
                        curr_g = line[1:].split()[0]
                        curr_s = []
                    else:
                        curr_s.append(line)
                if curr_g:
                    self.gene_seqs[curr_g] = "".join(curr_s)

        # 6. Deep Fusion Model & Calibration
        logger.info(f"Loading Fusion ResNet checkpoint from {checkpoint_path}...")
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.continuous_cols = ckpt["continuous_cols"]
        self.binary_cols = ckpt["binary_cols"]
        self.scaler_mean = np.array(ckpt["scaler_mean"], dtype=np.float32)
        self.scaler_scale = np.array(ckpt["scaler_scale"], dtype=np.float32)
        self.temperature = float(ckpt["calibration_temperature"])

        tabular_dim = len(self.continuous_cols) + len(self.binary_cols)
        self.model = FusionResNet(
            tabular_dim=tabular_dim,
            num_variant_classes=12,
            num_syndromes=15,
            use_gene_embedding=False,
            esm_emb_dim=480,
            esm_proj_dim=64,
            hidden_dim=256,
            dropout=0.0,
        ).to(self.device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        logger.info(f"Model ready (T={self.temperature:.4f}). Pipeline fully initialized.")

    def parse_vcf(
        self,
        vcf_path: Path,
        min_dp: int = 10,
        min_gq: int = 20,
        filter_pass_only: bool = True,
    ) -> List[VariantRecord]:
        """Parses and filters VCF file, checking GRCh38 compliance."""
        logger.info(f"Parsing VCF: {vcf_path}...")
        records = []
        is_gz = str(vcf_path).endswith(".gz")
        opener = gzip.open if is_gz else open

        header_lines = []
        is_grch38 = False
        is_hg19 = False

        with opener(vcf_path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("##"):
                    header_lines.append(line)
                    lower_line = line.lower()
                    if "grch38" in lower_line or "hg38" in lower_line:
                        is_grch38 = True
                    elif "hg19" in lower_line or "grch37" in lower_line or "b37" in lower_line:
                        is_hg19 = True
                    continue

                if line.startswith("#CHROM"):
                    # Check reference assembly
                    if is_hg19 and not is_grch38:
                        raise ValueError(
                            f"VCF header indicates reference assembly hg19 / GRCh37 ({vcf_path}). "
                            "GenomeDL v3 requires GRCh38 coordinates. Please lift over your VCF to GRCh38 "
                            "using UCSC liftOver or Picard LiftoverVcf before running inference."
                        )
                    continue

                fields = line.split("\t")
                if len(fields) < 5:
                    continue

                chrom = fields[0].replace("chr", "")
                pos = int(fields[1])
                var_id = fields[2]
                ref = fields[3].upper()
                alts = [a.upper() for a in fields[4].split(",")]
                qual = float(fields[5]) if fields[5] != "." else 100.0
                flt = fields[6]
                info_str = fields[7] if len(fields) > 7 else ""
                fmt = fields[8] if len(fields) > 8 else ""
                sample_data = fields[9] if len(fields) > 9 else ""

                if filter_pass_only and flt not in ["PASS", "."]:
                    continue

                # Parse genotype, DP, GQ
                gt = "0/1"
                dp = 30
                gq = 99
                if fmt and sample_data:
                    fmt_keys = fmt.split(":")
                    sample_vals = sample_data.split(":")
                    fmt_dict = dict(zip(fmt_keys, sample_vals))
                    gt = fmt_dict.get("GT", "0/1")
                    dp = int(fmt_dict.get("DP", 30)) if fmt_dict.get("DP", ".").isdigit() else 30
                    gq = int(fmt_dict.get("GQ", 99)) if fmt_dict.get("GQ", ".").isdigit() else 99

                if dp < min_dp or gq < min_gq:
                    continue

                is_hom_alt = gt in ["1/1", "1|1"]

                # Parse annotations (CSQ or ANN or local panel coordinate match)
                for alt in alts:
                    parsed_ann = self._parse_annotations(info_str, chrom, pos, ref, alt)
                    gene = parsed_ann.get("gene", "")
                    if gene not in self.panel:
                        # Skip variants not in our 130 panel genes
                        continue

                    vr = VariantRecord(
                        chrom=chrom,
                        pos=pos,
                        id=var_id,
                        ref=ref,
                        alt=alt,
                        qual=qual,
                        filter_status=flt,
                        gene=gene,
                        transcript=parsed_ann.get("transcript", ""),
                        consequence=parsed_ann.get("consequence", ""),
                        ref_aa=parsed_ann.get("ref_aa", ""),
                        alt_aa=parsed_ann.get("alt_aa", ""),
                        aa_pos=parsed_ann.get("aa_pos", -1),
                        hgvs_c=parsed_ann.get("hgvs_c", ""),
                        hgvs_p=parsed_ann.get("hgvs_p", ""),
                        af=parsed_ann.get("af", 0.0),
                        genotype=gt,
                        dp=dp,
                        gq=gq,
                        is_biallelic=is_hom_alt,
                        spliceai_max=float(parsed_ann.get("spliceai_max", 0.0)),
                    )
                    records.append(vr)

        # Check for compound heterozygosity in recessive genes
        gene_variants: Dict[str, List[VariantRecord]] = {}
        for r in records:
            gene_variants.setdefault(r.gene, []).append(r)

        for gene, v_list in gene_variants.items():
            if gene in RECESSIVE_GENES and len(v_list) >= 2:
                for v in v_list:
                    v.is_biallelic = True

        logger.info(f"Extracted {len(records)} candidate panel variants across {len(gene_variants)} genes.")
        return records

    def _parse_annotations(self, info_str: str, chrom: str, pos: int, ref: str, alt: str) -> Dict[str, Union[str, int, float]]:
        """Extracts gene, consequence, protein change, and AF from INFO string."""
        ann = {
            "gene": "", "transcript": "", "consequence": "",
            "ref_aa": "", "alt_aa": "", "aa_pos": -1,
            "hgvs_c": "", "hgvs_p": "", "af": 0.0,
            "spliceai_max": 0.0,
        }

        # Check gnomAD / AF in INFO
        m_af = re.search(r"(?:AF|gnomAD_AF|AF_raw)=([0-9.eE-]+)", info_str)
        if m_af:
            try:
                ann["af"] = float(m_af.group(1))
            except ValueError:
                pass

        # Check SpliceAI delta scores in INFO (Standard: SpliceAI=ALLELE|SYMBOL|DS_AG|DS_AL|DS_DG|DS_DL|...)
        m_sp = re.search(r"SpliceAI=([^;]+)", info_str)
        if m_sp:
            for entry in m_sp.group(1).split(","):
                sp_parts = entry.split("|")
                if len(sp_parts) >= 6:
                    try:
                        scores = [float(sp_parts[2]), float(sp_parts[3]), float(sp_parts[4]), float(sp_parts[5])]
                        ann["spliceai_max"] = max(ann["spliceai_max"], max(scores))
                    except (ValueError, IndexError):
                        pass

        # Standalone SpliceAI tags (e.g. DS_MAX=0.85, SpliceAI_max=0.85)
        m_sp_max = re.search(r"(?:DS_MAX|SpliceAI_DS_max|SpliceAI_max)=([0-9.]+)", info_str, re.IGNORECASE)
        if m_sp_max:
            try:
                ann["spliceai_max"] = max(ann["spliceai_max"], float(m_sp_max.group(1)))
            except ValueError:
                pass

        # Check VEP CSQ tag
        m_csq = re.search(r"CSQ=([^;]+)", info_str)
        if m_csq:
            # CSQ fields typically: Allele|Consequence|IMPACT|SYMBOL|Gene|Feature_type|Feature|BIOTYPE|EXON|INTRON|HGVSc|HGVSp|cDNA_position|CDS_position|Protein_position|Amino_acids|Codons|...
            for entry in m_csq.group(1).split(","):
                csq_parts = entry.split("|")
                if len(csq_parts) >= 4:
                    symbol = csq_parts[3].upper()
                    if symbol in self.panel:
                        ann["gene"] = symbol
                        ann["consequence"] = csq_parts[1] if len(csq_parts) > 1 else ""
                        ann["transcript"] = csq_parts[6] if len(csq_parts) > 6 else ""
                        ann["hgvs_c"] = csq_parts[10] if len(csq_parts) > 10 else ""
                        ann["hgvs_p"] = csq_parts[11] if len(csq_parts) > 11 else ""
                        # Amino acids (e.g. C/G)
                        if len(csq_parts) > 15 and "/" in csq_parts[15]:
                            aa_pair = csq_parts[15].split("/")
                            ann["ref_aa"] = aa_pair[0]
                            ann["alt_aa"] = aa_pair[1]
                        # Protein position
                        if len(csq_parts) > 14 and csq_parts[14].isdigit():
                            ann["aa_pos"] = int(csq_parts[14])
                        break

        # Fallback: check ClinVar lookup table for this exact (chrom, pos, ref, alt)
        key = (chrom, pos, ref, alt)
        if not ann["gene"] and key in self.clinvar_index:
            c_info = self.clinvar_index[key]
            ann["gene"] = c_info["gene"]
            ann["consequence"] = c_info["variant_class"]

        return ann

    def evaluate_variant(self, var: VariantRecord) -> VariantVerdict:
        """Evaluates a single variant through the 8-stage decision tree."""
        # Validate gene symbol
        if var.gene not in self.panel:
            raise ValueError(f"Gene '{var.gene}' not in 130-gene panel. Known panel genes: {sorted(list(self.panel.keys()))[:5]}...")

        # 1. Common variant filter
        if var.af > 0.01:
            return VariantVerdict(
                variant=var,
                tier="Benign",
                evidence_source=f"Common polymorphism (gnomAD AF = {var.af * 100:.2f}% > 1.0%)",
                calibrated_score=0.001,
                clinical_notes="High population frequency in gnomAD indicates benign polymorphism.",
            )

        # 2. ClinVar lookup with stars >= 2
        key = (var.chrom, var.pos, var.ref, var.alt)
        if key in self.clinvar_index:
            c_entry = self.clinvar_index[key]
            stars = c_entry["stars"]
            sig = c_entry["clinical_significance"].lower()
            if stars >= 2:
                tier = "Pathogenic" if "pathogenic" in sig and "benign" not in sig else "Benign"
                if "likely pathogenic" in sig:
                    tier = "Likely Pathogenic"
                elif "likely benign" in sig:
                    tier = "Likely Benign"
                elif "uncertain" in sig or "vus" in sig:
                    tier = "VUS"

                verdict = VariantVerdict(
                    variant=var,
                    tier=tier,
                    evidence_source=f"ClinVar lookup ({stars} stars: {c_entry['review_status']})",
                    calibrated_score=0.999 if tier == "Pathogenic" else (0.001 if tier == "Benign" else 0.5),
                    clinical_notes=f"Curated clinical consensus from ClinVar submission (Variation ID {c_entry['variation_id']}).",
                )
                self._attach_risk_estimates(verdict)
                return verdict

        # 3. Rule-based Loss-of-Function & Splicing Safeguards
        csq = var.consequence.lower()

        # 3A. Canonical Loss-of-Function (Frameshift, canonical splice donor/acceptor +/-1,2)
        if any(term in csq for term in ["frameshift", "splice_acceptor", "splice_donor", "splice_canonical"]):
            verdict = VariantVerdict(
                variant=var,
                tier="Likely Pathogenic",
                evidence_source="Rule-based (predicted loss-of-function truncation/canonical splice disruption)",
                calibrated_score=0.950,
                clinical_notes="Canonical splice-site disruption or frameshift truncation in established tumor suppressor.",
            )
            self._attach_risk_estimates(verdict)
            return verdict

        # 3B. SpliceAI Deep Splicing Safeguard (Novel deep-intronic / synonymous splice disruptors)
        # Bypasses protein ResNet model to avoid false-negative classification of RNA-altering variants
        if var.spliceai_max >= 0.50:
            verdict = VariantVerdict(
                variant=var,
                tier="Likely Pathogenic",
                evidence_source=f"Rule-based (SpliceAI high-confidence splice alteration, DS_MAX = {var.spliceai_max:.3f} >= 0.50)",
                calibrated_score=0.950,
                clinical_notes=(
                    f"High-confidence cryptic splice alteration predicted by SpliceAI (DS_MAX={var.spliceai_max:.3f}). "
                    "Bypasses protein-folding model to prevent false-negative misclassification of non-coding/synonymous splice disruptor."
                ),
            )
            self._attach_risk_estimates(verdict)
            return verdict
        elif var.spliceai_max >= 0.20:
            verdict = VariantVerdict(
                variant=var,
                tier="VUS",
                evidence_source=f"SpliceAI moderate splice disruption (DS_MAX = {var.spliceai_max:.3f} >= 0.20)",
                calibrated_score=0.500,
                clinical_notes=(
                    f"Moderate splice alteration predicted by SpliceAI (DS_MAX={var.spliceai_max:.3f}). "
                    "Functional RNA assay or RT-PCR transcript analysis recommended to confirm aberrant splicing."
                ),
            )
            self._attach_risk_estimates(verdict)
            return verdict

        # 3C. Extended Splice Region Boundary Warning (Unclassified non-canonical splice variant without SpliceAI)
        if "splice_region" in csq and not var.ref_aa and not var.alt_aa:
            verdict = VariantVerdict(
                variant=var,
                tier="VUS",
                evidence_source="Splice region boundary (unclassified non-canonical splice variant)",
                calibrated_score=0.400,
                clinical_notes="Located in non-canonical splice region (-3 to +8 bp / polypyrimidine tract). Requires RNA sequencing or SpliceAI annotation to confirm splicing impact.",
            )
            self._attach_risk_estimates(verdict)
            return verdict

        # 4. Deep Multimodal Model Inference for SNVs
        score = self._run_model_inference(var)
        if score >= 0.90:
            tier = "Pathogenic"
        elif score >= 0.70:
            tier = "Likely Pathogenic"
        elif score >= 0.30:
            tier = "VUS"
        elif score >= 0.10:
            tier = "Likely Benign"
        else:
            tier = "Benign"

        verdict = VariantVerdict(
            variant=var,
            tier=tier,
            evidence_source="GenomeDL v3 ESM-Augmented Fusion ResNet",
            calibrated_score=score,
            clinical_notes=f"Calibrated probability P(pathogenic) = {score:.4f}.",
        )
        self._attach_risk_estimates(verdict)
        return verdict

    def _run_model_inference(self, var: VariantRecord) -> float:
        """Extracts features, runs FusionResNet, and returns calibrated probability."""
        gene_info = self.panel.get(var.gene, {})
        syndrome = gene_info.get("syndrome_group", "Other")
        syndrome_id = self.syndrome_to_id.get(syndrome, 0)
        prot_len = gene_info.get("protein_length", 1000)
        strand = gene_info.get("strand", "+")

        # Variant class
        csq = var.consequence.lower()
        if "missense" in csq:
            vclass = "missense"
        elif "synonymous" in csq:
            vclass = "synonymous"
        elif "stop_gained" in csq or "nonsense" in csq:
            vclass = "nonsense"
        elif "splice_donor" in csq or "splice_acceptor" in csq:
            vclass = "splice_canonical"
        elif "splice" in csq:
            vclass = "splice_region"
        elif "intron" in csq:
            vclass = "intronic"
        elif "5_prime_utr" in csq or "utr5" in csq:
            vclass = "utr5"
        elif "3_prime_utr" in csq or "utr3" in csq:
            vclass = "utr3"
        elif "start_lost" in csq:
            vclass = "start_lost"
        elif "stop_lost" in csq:
            vclass = "stop_lost"
        else:
            vclass = "other_noncoding"
        vclass_id = CLASS_TO_ID.get(vclass, 10)

        # Continuous biochemical features
        has_prot = 1 if vclass in ["missense", "nonsense", "start_lost", "stop_lost"] else 0
        rel_pos = float(var.aa_pos) / float(prot_len) if prot_len > 0 and var.aa_pos > 0 else 0.5

        from rdkit.Chem import DataStructs

        ref_1 = var.ref_aa
        alt_1 = var.alt_aa

        if vclass == "start_lost":
            grantham = 215.0
            tanimoto = 1.0
            if alt_1 in AA_PROPS:
                dmw = abs(AA_PROPS["M"]["mw"] - AA_PROPS[alt_1]["mw"])
                dcharge = abs(AA_PROPS["M"]["charge"] - AA_PROPS[alt_1]["charge"])
                dlogp = abs(AA_PROPS["M"]["logp"] - AA_PROPS[alt_1]["logp"])
                dtpsa = abs(AA_PROPS["M"]["tpsa"] - AA_PROPS[alt_1]["tpsa"])
            else:
                dmw, dcharge, dlogp, dtpsa = 50.0, 0.0, 1.0, 20.0
        elif ref_1 in AA_PROPS and alt_1 in AA_PROPS:
            grantham = float(GRANTHAM.get((ref_1, alt_1), 0.0))
            sim = float(DataStructs.TanimotoSimilarity(AA_PROPS[ref_1]["fp"], AA_PROPS[alt_1]["fp"]))
            tanimoto = 1.0 - sim
            dmw = abs(AA_PROPS[ref_1]["mw"] - AA_PROPS[alt_1]["mw"])
            dcharge = abs(AA_PROPS[ref_1]["charge"] - AA_PROPS[alt_1]["charge"])
            dlogp = abs(AA_PROPS[ref_1]["logp"] - AA_PROPS[alt_1]["logp"])
            dtpsa = abs(AA_PROPS[ref_1]["tpsa"] - AA_PROPS[alt_1]["tpsa"])
        else:
            grantham = 0.0
            tanimoto = 0.0
            dmw = 0.0
            dcharge = 0.0
            dlogp = 0.0
            dtpsa = 0.0

        raw_cont = np.array([rel_pos, grantham, dmw, dcharge, dlogp, dtpsa, tanimoto], dtype=np.float32)
        scaled_cont = (raw_cont - self.scaler_mean) / self.scaler_scale

        # Binary features
        is_trans = 1 if (var.ref, var.alt) in [("A", "G"), ("G", "A"), ("C", "T"), ("T", "C")] else 0
        bin_dict = {f"ref_{b}": 1 if var.ref == b else 0 for b in ["A", "C", "G", "T"]}
        bin_dict.update({f"alt_{b}": 1 if var.alt == b else 0 for b in ["A", "C", "G", "T"]})

        comp = {"A": "T", "T": "A", "C": "G", "G": "C"}
        cs_ref = var.ref if strand == "+" else comp.get(var.ref, var.ref)
        cs_alt = var.alt if strand == "+" else comp.get(var.alt, var.alt)
        bin_dict.update({f"cs_ref_{b}": 1 if cs_ref == b else 0 for b in ["A", "C", "G", "T"]})
        bin_dict.update({f"cs_alt_{b}": 1 if cs_alt == b else 0 for b in ["A", "C", "G", "T"]})

        bin_list = [1, has_prot, is_trans] + [bin_dict[col] for col in BINARY_COLS if col not in ["on_primary_transcript", "has_protein_change", "is_transition"]]
        raw_bin = np.array(bin_list, dtype=np.float32)

        tabular_feat = np.hstack([scaled_cont, raw_bin])

        # ESM features
        esm_key = (var.gene, var.aa_pos)
        if esm_key in self.esm_cache:
            log_p_dict, emb_vec = self.esm_cache[esm_key]
            alt_k = "*" if var.alt_aa in ["Ter", "*"] else var.alt_aa
            ref_k = "*" if var.ref_aa in ["Ter", "*"] else var.ref_aa
            if alt_k in log_p_dict and ref_k in log_p_dict:
                llr = log_p_dict[alt_k] - log_p_dict[ref_k]
                avail = 1.0
                emb = emb_vec
            else:
                llr, avail, emb = 0.0, 0.0, np.zeros(480, dtype=np.float32)
        else:
            llr, avail, emb = 0.0, 0.0, np.zeros(480, dtype=np.float32)

        # PyTorch Tensors
        with torch.no_grad():
            t_tab = torch.tensor(tabular_feat, device=self.device, dtype=torch.float32).unsqueeze(0)
            t_vc = torch.tensor([vclass_id], device=self.device, dtype=torch.long)
            t_sy = torch.tensor([syndrome_id], device=self.device, dtype=torch.long)
            t_llr = torch.tensor([[llr]], device=self.device, dtype=torch.float32)
            t_avail = torch.tensor([[avail]], device=self.device, dtype=torch.float32)
            t_emb = torch.tensor(emb, device=self.device, dtype=torch.float32).unsqueeze(0)

            raw_logit = self.model(t_tab, t_vc, t_sy, t_llr, t_avail, t_emb).item()
            calibrated_logit = raw_logit / self.temperature
            calibrated_prob = float(1.0 / (1.0 + np.exp(-calibrated_logit)))

        return calibrated_prob

    def _attach_risk_estimates(self, verdict: VariantVerdict):
        """Attaches published penetrance ranges and handles recessive zygosity."""
        gene = verdict.variant.gene

        # Handle recessive zygosity (MUTYH, NTHL1)
        if gene in RECESSIVE_GENES:
            if not verdict.variant.is_biallelic:
                verdict.tier = "Carrier (Recessive)"
                verdict.clinical_notes += (
                    f" Single heterozygous allele in autosomal recessive gene {gene}. "
                    "Biallelic mutation is required for elevated cancer risk. Not considered high penetrance carrier."
                )
                return

        # Lookup in curated penetrance table
        if not self.penetrance_df.empty:
            gene_rows = self.penetrance_df[self.penetrance_df["gene"] == gene]
            for _, r in gene_rows.iterrows():
                outcome = str(r["outcome"])
                entry = {
                    "outcome": outcome,
                    "risk_low_pct": float(r["risk_low_pct"]),
                    "risk_high_pct": float(r["risk_high_pct"]),
                    "up_to_age": int(r["up_to_age"]),
                    "source": str(r["source"]),
                    "evidence_level": str(r["evidence_level"]),
                }
                # Separate non-cancer features
                if any(k in outcome.lower() for k in ["pneumothorax", "leiomyomatosis", "thrombocytopenia", "aplastic anemia", "bone marrow"]):
                    verdict.other_syndrome_features.append(entry)
                else:
                    verdict.cancer_risks.append(entry)

    def generate_report(self, verdicts: List[VariantVerdict], output_json: Optional[Path] = None) -> str:
        """Generates comprehensive human-readable report and structured JSON."""
        lines = []
        lines.append("=" * 80)
        lines.append("GENOMEDL v3: CLINICAL GENOMICS INFERENCE REPORT")
        lines.append("=" * 80)
        lines.append(f"TOTAL PANEL VARIANTS EVALUATED: {len(verdicts)}")

        actionable_verdicts = [v for v in verdicts if v.tier in ["Pathogenic", "Likely Pathogenic"]]
        lines.append(f"ACTIONABLE (PATHOGENIC / LIKELY PATHOGENIC): {len(actionable_verdicts)}")
        lines.append("-" * 80)

        # Aggregate cumulative cancer risks by organ / cancer type
        cancer_risk_map = {}
        for v in actionable_verdicts:
            gene = v.variant.gene
            for r in v.cancer_risks:
                outcome = r["outcome"]
                if outcome not in cancer_risk_map:
                    cancer_risk_map[outcome] = {
                        "genes": set(),
                        "lows": [],
                        "highs": [],
                        "ages": [],
                        "sources": [],
                    }
                cancer_risk_map[outcome]["genes"].add(gene)
                cancer_risk_map[outcome]["lows"].append(r["risk_low_pct"])
                cancer_risk_map[outcome]["highs"].append(r["risk_high_pct"])
                cancer_risk_map[outcome]["ages"].append(r["up_to_age"])
                cancer_risk_map[outcome]["sources"].append(f"{r['source']} ({gene})")

        # Top-level executive risk table
        cancer_summary_list = []
        if cancer_risk_map:
            lines.append("")
            lines.append("=" * 80)
            lines.append("EXECUTIVE SUMMARY: PREDICTED LIFETIME CANCER RISKS BY ORGAN / TYPE")
            lines.append("=" * 80)
            lines.append(f"{'CANCER TYPE':<32} | {'ESTIMATED LIFETIME RISK':<23} | {'GENE(S) INVOLVED':<16} | {'HORIZON'}")
            lines.append("-" * 80)
            
            # Sort by highest upper bound risk descending
            sorted_outcomes = sorted(cancer_risk_map.items(), key=lambda x: max(x[1]["highs"]), reverse=True)
            for outcome, data in sorted_outcomes:
                genes_str = ", ".join(sorted(data["genes"]))
                # Primary carrier risk range from dominant driver gene
                risk_str = f"{max(data['lows']):.0f}% - {max(data['highs']):.0f}%"
                age_str = f"Up to age {max(data['ages'])}"
                lines.append(f"{outcome:<32} | {risk_str:<23} | {genes_str:<16} | {age_str}")
                cancer_summary_list.append({
                    "cancer_type": outcome,
                    "lifetime_risk_range": risk_str,
                    "genes": sorted(list(data["genes"])),
                    "up_to_age": max(data["ages"]),
                })
            lines.append("-" * 80)
            lines.append("")

        for i, v in enumerate(verdicts, 1):
            var = v.variant
            lines.append(f"\nVARIANT #{i}: {var.gene} chr{var.chrom}:{var.pos} {var.ref}>{var.alt}")
            lines.append(f"  HGVS / Consequence: {var.hgvs_p or var.hgvs_c or var.consequence}")
            lines.append(f"  Genotype:           {var.genotype} (DP={var.dp}, GQ={var.gq})")
            lines.append(f"  VERDICT TIER:       [{v.tier.upper()}]")
            lines.append(f"  Evidence Source:    {v.evidence_source}")
            if v.calibrated_score is not None:
                lines.append(f"  Calibrated Model P: {v.calibrated_score:.4f}")
            if v.clinical_notes:
                lines.append(f"  Clinical Notes:     {v.clinical_notes}")

            if v.tier in ["Pathogenic", "Likely Pathogenic"]:
                lines.append("  >>> Published Lifetime Carrier Cancer Risks (Source: data/penetrance.csv):")
                if v.cancer_risks:
                    for r in v.cancer_risks:
                        lines.append(
                            f"      - {r['outcome']}: {r['risk_low_pct']:.0f}% - {r['risk_high_pct']:.0f}% (up to age {r['up_to_age']}) "
                            f"[Source: {r['source']}, Evidence: {r['evidence_level']}]"
                        )
                else:
                    lines.append("      - No curated lifetime risk estimate available for this gene.")

                if v.other_syndrome_features:
                    lines.append("  >>> Other Syndromic Features (Non-Cancer):")
                    for r in v.other_syndrome_features:
                        lines.append(
                            f"      - {r['outcome']}: {r['risk_low_pct']:.0f}% - {r['risk_high_pct']:.0f}% (up to age {r['up_to_age']}) "
                            f"[Source: {r['source']}]"
                        )

        lines.append("\n" + "=" * 80)
        lines.append("MANDATORY CLINICAL DISCLAIMER:")
        lines.append(f"  {DISCLAIMER_TEXT}")
        lines.append("=" * 80 + "\n")

        report_text = "\n".join(lines)

        if output_json:
            json_data = {
                "total_variants": len(verdicts),
                "actionable_count": len(actionable_verdicts),
                "cancer_risk_summary": cancer_summary_list,
                "verdicts": [
                    {
                        "gene": v.variant.gene,
                        "chrom": v.variant.chrom,
                        "pos": v.variant.pos,
                        "ref": v.variant.ref,
                        "alt": v.variant.alt,
                        "tier": v.tier,
                        "evidence_source": v.evidence_source,
                        "calibrated_score": v.calibrated_score,
                        "cancer_risks": v.cancer_risks,
                        "other_syndrome_features": v.other_syndrome_features,
                        "notes": v.clinical_notes,
                    }
                    for v in verdicts
                ],
                "disclaimer": DISCLAIMER_TEXT,
            }
            output_json.parent.mkdir(parents=True, exist_ok=True)
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(json_data, f, indent=2)
            logger.info(f"Saved structured JSON report to {output_json}")

        return report_text


def parse_args():
    parser = argparse.ArgumentParser(description="GenomeDL v3 Clinical VCF Pipeline")
    parser.add_argument("vcf_path", type=Path, help="Path to input VCF (.vcf or .vcf.gz)")
    parser.add_argument("--out_report", type=Path, default=Path("reports/vcf_report.txt"))
    parser.add_argument("--out_json", type=Path, default=Path("reports/vcf_report.json"))
    parser.add_argument("--out_pdf", type=Path, default=Path("reports/vcf_report.pdf"))
    parser.add_argument("--no_pdf", action="store_true", help="Disable automatic PDF generation")
    parser.add_argument("--min_dp", type=int, default=10)
    parser.add_argument("--min_gq", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = VCFPipeline()
    variants = pipeline.parse_vcf(args.vcf_path, min_dp=args.min_dp, min_gq=args.min_gq)
    verdicts = [pipeline.evaluate_variant(v) for v in variants]
    rep = pipeline.generate_report(verdicts, output_json=args.out_json)
    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_report, "w", encoding="utf-8") as f:
        f.write(rep)
    print(rep)

    if not args.no_pdf and args.out_pdf:
        try:
            from src.v3.generate_pdf_report import generate_pdf_report
            sample_name = args.vcf_path.stem.upper()
            pdf_path = generate_pdf_report(args.out_json, args.out_pdf, sample_id=sample_name)
            logger.info(f"Clinical PDF report successfully generated: {pdf_path}")
        except Exception as e:
            logger.warning(f"Could not generate PDF report: {e}")
