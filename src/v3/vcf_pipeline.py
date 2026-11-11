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
DEFAULT_PANEL_COORDS = DATA_DIR / "panel_coordinates.json"
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
    af: Optional[float] = None
    genotype: str = "0/1"
    dp: Optional[int] = None
    gq: Optional[int] = None
    is_biallelic: bool = False
    phase_set: Optional[str] = None
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
        panel_coords_path: Path = DEFAULT_PANEL_COORDS,
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

        # 1b. Panel genomic coordinate intervals
        self.panel_coords = {}
        if panel_coords_path.exists():
            self.panel_coords = json.load(open(panel_coords_path, "r", encoding="utf-8"))
            logger.info(f"Loaded genomic coordinate intervals for {len(self.panel_coords)} panel genes.")

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
        self.temperature = float(ckpt.get("calibration_temperature", 1.0))
        self.calibration_scale = float(ckpt.get("calibration_scale", 1.0 / self.temperature if self.temperature > 0 else 1.0))
        self.calibration_bias = float(ckpt.get("calibration_bias", 0.0))

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

        self.ensemble_models = []
        if "ensemble_state_dicts" in ckpt and ckpt["ensemble_state_dicts"]:
            for s_dict in ckpt["ensemble_state_dicts"]:
                m = FusionResNet(
                    tabular_dim=tabular_dim,
                    num_variant_classes=12,
                    num_syndromes=15,
                    use_gene_embedding=False,
                    esm_emb_dim=480,
                    esm_proj_dim=64,
                    hidden_dim=256,
                    dropout=0.0,
                ).to(self.device)
                m.load_state_dict(s_dict)
                m.eval()
                self.ensemble_models.append(m)
            logger.info(f"Loaded ensemble of {len(self.ensemble_models)} FusionResNet seeds.")

        logger.info(f"Model ready (T={self.temperature:.4f}, scale={self.calibration_scale:.4f}, bias={self.calibration_bias:.4f}). Pipeline fully initialized.")

    @staticmethod
    def _validate_vcf_reference(header_lines: List[str], vcf_path: Path) -> None:
        """Validates that the VCF header positively identifies GRCh38 / hg38."""
        is_grch38 = False
        is_hg19 = False
        for line in header_lines:
            lower_line = line.lower()
            if (
                "grch38" in lower_line
                or "hg38" in lower_line
                or "hs38" in lower_line
                or "gca_000001405.15" in lower_line
                or re.search(r"contig=<[^>]*id=(?:chr)?1\b[^>]*length=248956422\b", lower_line)
            ):
                is_grch38 = True
            elif (
                "hg19" in lower_line
                or "grch37" in lower_line
                or "b37" in lower_line
                or "hs37" in lower_line
                or re.search(r"contig=<[^>]*id=(?:chr)?1\b[^>]*length=249250621\b", lower_line)
            ):
                is_hg19 = True

        if is_hg19 and not is_grch38:
            raise ValueError(
                f"VCF header indicates reference assembly hg19 / GRCh37 ({vcf_path}). "
                "GenomeDL v3 requires GRCh38 coordinates. Please lift over your VCF to GRCh38 "
                "using UCSC liftOver or Picard LiftoverVcf before running inference."
            )
        if not is_grch38:
            raise ValueError(
                f"VCF header does not positively identify GRCh38 / hg38 ({vcf_path}). "
                "GenomeDL v3 requires an explicit GRCh38/hg38 reference or contig header."
            )

    @staticmethod
    def _parse_gt_alleles(gt: Optional[str]) -> Optional[List[Optional[int]]]:
        """Parses a VCF GT field into integer allele indices (or None for missing)."""
        if gt is None:
            return None
        cleaned = gt.strip()
        if not cleaned or cleaned in (".", "./.", ".|."):
            return None
        tokens = re.split(r"[/|]", cleaned)
        alleles: List[Optional[int]] = []
        for tok in tokens:
            alleles.append(int(tok) if tok.isdigit() else None)
        return alleles

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
        header_validated = False

        with opener(vcf_path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("##"):
                    header_lines.append(line)
                    continue

                if line.startswith("#CHROM"):
                    self._validate_vcf_reference(header_lines, vcf_path)
                    header_validated = True
                    continue

                if not header_validated:
                    self._validate_vcf_reference(header_lines, vcf_path)
                    header_validated = True

                fields = line.split("\t")
                if len(fields) < 5:
                    continue

                chrom = fields[0].replace("chr", "")
                pos = int(fields[1])
                var_id = fields[2]
                ref = fields[3].upper()
                alts = [a.upper() for a in fields[4].split(",")]
                qual = float(fields[5]) if len(fields) > 5 and fields[5] not in (".", "") else 100.0
                flt = fields[6] if len(fields) > 6 else "."
                info_str = fields[7] if len(fields) > 7 else ""
                fmt = fields[8] if len(fields) > 8 else ""
                sample_data = fields[9] if len(fields) > 9 else ""

                if filter_pass_only and flt != "PASS":
                    continue

                # Parse genotype, depth, quality, and phase set without inventing missing quality values
                gt: Optional[str] = None
                dp: Optional[int] = None
                gq: Optional[int] = None
                phase_set: Optional[str] = None
                if fmt and sample_data:
                    fmt_dict = dict(zip(fmt.split(":"), sample_data.split(":")))
                    raw_gt = fmt_dict.get("GT", "").strip()
                    if raw_gt and raw_gt != ".":
                        gt = raw_gt
                    raw_dp = fmt_dict.get("DP", "").strip()
                    if raw_dp.isdigit():
                        dp = int(raw_dp)
                    raw_gq = fmt_dict.get("GQ", "").strip()
                    if raw_gq.isdigit():
                        gq = int(raw_gq)
                    raw_ps = fmt_dict.get("PS", "").strip()
                    if raw_ps and raw_ps != ".":
                        phase_set = raw_ps

                if min_dp > 0 and (dp is None or dp < min_dp):
                    continue
                if min_gq > 0 and (gq is None or gq < min_gq):
                    continue

                gt_alleles = self._parse_gt_alleles(gt)
                if gt is None or gt_alleles is None or not any(a is not None and a > 0 for a in gt_alleles):
                    continue

                # Parse annotations per called ALT allele
                for alt_idx, alt in enumerate(alts):
                    target_allele = alt_idx + 1
                    if target_allele not in gt_alleles:
                        continue

                    is_hom_alt = len(gt_alleles) >= 2 and all(a == target_allele for a in gt_alleles)
                    parsed_ann = self._parse_annotations(
                        info_str, chrom, pos, ref, alt, alt_idx=alt_idx, num_alts=len(alts)
                    )
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
                        af=parsed_ann.get("af"),
                        genotype=gt,
                        dp=dp,
                        gq=gq,
                        is_biallelic=is_hom_alt,
                        phase_set=phase_set,
                        spliceai_max=float(parsed_ann.get("spliceai_max", 0.0)),
                    )
                    records.append(vr)

        if not header_validated and header_lines:
            self._validate_vcf_reference(header_lines, vcf_path)
        elif not header_validated:
            self._validate_vcf_reference([], vcf_path)

        gene_variants: Dict[str, List[VariantRecord]] = {}
        for r in records:
            gene_variants.setdefault(r.gene, []).append(r)

        logger.info(f"Extracted {len(records)} candidate panel variants across {len(gene_variants)} genes.")
        return records

    def _parse_annotations(
        self,
        info_str: str,
        chrom: str,
        pos: int,
        ref: str,
        alt: str,
        alt_idx: int = 0,
        num_alts: int = 1,
    ) -> Dict[str, Optional[Union[str, int, float]]]:
        """Extracts allele-specific gene, consequence, protein change, SpliceAI, and AF from INFO."""
        ann: Dict[str, Optional[Union[str, int, float]]] = {
            "gene": "",
            "transcript": "",
            "consequence": "",
            "ref_aa": "",
            "alt_aa": "",
            "aa_pos": -1,
            "hgvs_c": "",
            "hgvs_p": "",
            "af": None,
            "spliceai_max": 0.0,
        }

        # Check gnomAD / AF in INFO (per-ALT allele, keeping missing distinct from 0.0)
        m_af = re.search(r"(?:^|;)(?:AF|gnomAD_AF|AF_raw)=([^;]+)", info_str)
        if m_af:
            af_vals = m_af.group(1).split(",")
            raw_af = None
            if alt_idx < len(af_vals):
                raw_af = af_vals[alt_idx].strip()
            elif num_alts == 1 and len(af_vals) == 1:
                raw_af = af_vals[0].strip()
            if raw_af and raw_af != ".":
                try:
                    ann["af"] = float(raw_af)
                except ValueError:
                    pass

        # Check SpliceAI delta scores in INFO matched to the evaluated ALT allele
        m_sp = re.search(r"(?:^|;)SpliceAI=([^;]+)", info_str)
        if m_sp:
            for entry in m_sp.group(1).split(","):
                sp_parts = entry.split("|")
                if len(sp_parts) >= 6:
                    sp_allele = sp_parts[0].strip().upper()
                    if sp_allele and sp_allele != ".":
                        if sp_allele != alt:
                            continue
                    elif num_alts > 1:
                        continue
                    try:
                        scores = [float(s) for s in sp_parts[2:6] if s not in ("", ".")]
                        if scores:
                            ann["spliceai_max"] = max(float(ann["spliceai_max"]), max(scores))
                    except (ValueError, IndexError):
                        pass

        # Standalone SpliceAI tags (single-ALT lines only to avoid cross-allele leakage)
        if num_alts == 1:
            m_sp_max = re.search(r"(?:^|;)(?:DS_MAX|SpliceAI_DS_max|SpliceAI_max)=([0-9.]+)", info_str, re.IGNORECASE)
            if m_sp_max:
                try:
                    ann["spliceai_max"] = max(float(ann["spliceai_max"]), float(m_sp_max.group(1)))
                except ValueError:
                    pass

        # Check VEP CSQ tag matching the evaluated ALT allele
        def _allele_matches(entry_allele: str) -> bool:
            ea = entry_allele.strip().upper()
            if not ea or ea == ".":
                return num_alts == 1
            if ea == alt:
                return True
            # VEP minimal representation trims common anchor base for indels
            if len(ref) != len(alt):
                if ea == "-" and len(alt) == 1:
                    return True
                if len(alt) > 1 and ea == alt[1:]:
                    return True
            return False

        m_csq = re.search(r"(?:^|;)CSQ=([^;]+)", info_str)
        if m_csq:
            # CSQ fields typically: Allele|Consequence|IMPACT|SYMBOL|Gene|Feature_type|Feature|BIOTYPE|EXON|INTRON|HGVSc|HGVSp|cDNA_position|CDS_position|Protein_position|Amino_acids|Codons|...
            for entry in m_csq.group(1).split(","):
                csq_parts = entry.split("|")
                if len(csq_parts) >= 4 and _allele_matches(csq_parts[0]):
                    symbol = csq_parts[3].upper()
                    if symbol in self.panel:
                        ann["gene"] = symbol
                        ann["consequence"] = csq_parts[1] if len(csq_parts) > 1 else ""
                        ann["transcript"] = csq_parts[6] if len(csq_parts) > 6 else ""
                        ann["hgvs_c"] = csq_parts[10] if len(csq_parts) > 10 else ""
                        ann["hgvs_p"] = csq_parts[11] if len(csq_parts) > 11 else ""
                        if len(csq_parts) > 15 and "/" in csq_parts[15]:
                            aa_pair = csq_parts[15].split("/")
                            ann["ref_aa"] = aa_pair[0]
                            ann["alt_aa"] = aa_pair[1]
                        if len(csq_parts) > 14 and csq_parts[14].isdigit():
                            ann["aa_pos"] = int(csq_parts[14])
                        break

        # Check SnpEff ANN tag matching the evaluated ALT allele
        if not ann["gene"]:
            m_ann = re.search(r"(?:^|;)ANN=([^;]+)", info_str)
            if m_ann:
                for entry in m_ann.group(1).split(","):
                    ann_parts = entry.split("|")
                    if len(ann_parts) >= 4 and _allele_matches(ann_parts[0]):
                        symbol = ann_parts[3].upper()
                        if symbol in self.panel:
                            ann["gene"] = symbol
                            ann["consequence"] = ann_parts[1] if len(ann_parts) > 1 else ""
                            ann["transcript"] = ann_parts[6] if len(ann_parts) > 6 else ""
                            ann["hgvs_c"] = ann_parts[9] if len(ann_parts) > 9 else ""
                            ann["hgvs_p"] = ann_parts[10] if len(ann_parts) > 10 else ""
                            break

        # Fallback 1: check ClinVar lookup table for this exact (chrom, pos, ref, alt)
        key = (chrom, pos, ref, alt)
        if not ann["gene"] and key in self.clinvar_index:
            c_info = self.clinvar_index[key]
            ann["gene"] = c_info["gene"]
            ann["consequence"] = c_info["variant_class"]

        # Fallback 2: unambiguous genomic coordinate interval match against 130 panel genes
        if not ann["gene"] and self.panel_coords:
            chrom_clean = chrom.replace("chr", "")
            matching_genes = [
                g_sym
                for g_sym, coord in self.panel_coords.items()
                if str(coord.get("chrom", "")).replace("chr", "") == chrom_clean
                and coord.get("start", 0) <= pos <= coord.get("end", 0)
            ]
            if len(matching_genes) == 1:
                ann["gene"] = matching_genes[0]
                ann["consequence"] = "intronic"
            elif len(matching_genes) > 1:
                logger.warning(
                    f"Ambiguous coordinate fallback at chr{chrom_clean}:{pos} "
                    f"matches multiple panel genes {sorted(matching_genes)}; skipping unannotated assignment."
                )

        return ann

    def evaluate_variant(self, var: VariantRecord) -> VariantVerdict:
        """Evaluates a single variant through the 8-stage decision tree."""
        return self._evaluate_variant_core(var, resolve_recessive=True)

    @staticmethod
    def _classify_clinvar_significance(sig_raw: str) -> tuple[str, float]:
        """Maps a ClinVar clinical_significance string cleanly to (tier, score)."""
        sig = sig_raw.lower()
        has_conflict = "conflicting" in sig or "uncertain" in sig or "vus" in sig
        has_path = bool(re.search(r"\bpathogenic\b", sig))
        has_ben = bool(re.search(r"\bbenign\b", sig))

        if has_conflict or (has_path and has_ben):
            return "VUS", 0.5
        if has_path:
            if "likely pathogenic" in sig:
                return "Likely Pathogenic", 0.95
            return "Pathogenic", 0.999
        if has_ben:
            if "likely benign" in sig:
                return "Likely Benign", 0.05
            return "Benign", 0.001
        return "VUS", 0.5

    def _evaluate_variant_core(self, var: VariantRecord, resolve_recessive: bool = True) -> VariantVerdict:
        """Core 8-stage evaluation, optionally deferring recessive zygosity resolution."""
        # Validate gene symbol
        if var.gene not in self.panel:
            raise ValueError(f"Gene '{var.gene}' not in 130-gene panel. Known panel genes: {sorted(list(self.panel.keys()))[:5]}...")

        verdict: Optional[VariantVerdict] = None

        # 1. Common variant filter
        if var.af is not None and var.af > 0.01:
            verdict = VariantVerdict(
                variant=var,
                tier="Benign",
                evidence_source=f"Common polymorphism (gnomAD AF = {var.af * 100:.2f}% > 1.0%)",
                calibrated_score=0.001,
                clinical_notes="High population frequency in gnomAD indicates benign polymorphism.",
            )

        # 2. ClinVar lookup with stars >= 2
        if verdict is None:
            key = (var.chrom, var.pos, var.ref, var.alt)
            if key in self.clinvar_index:
                c_entry = self.clinvar_index[key]
                stars = c_entry["stars"]
                if stars >= 2:
                    tier, score = self._classify_clinvar_significance(c_entry["clinical_significance"])
                    verdict = VariantVerdict(
                        variant=var,
                        tier=tier,
                        evidence_source=f"ClinVar lookup ({stars} stars: {c_entry['review_status']})",
                        calibrated_score=score,
                        clinical_notes=f"Curated clinical consensus from ClinVar submission (Variation ID {c_entry['variation_id']}).",
                    )

        # 3. Rule-based Loss-of-Function & Splicing Safeguards
        csq = var.consequence.lower()

        # 3A. Canonical Loss-of-Function (Frameshift, canonical splice donor/acceptor +/-1,2)
        if verdict is None and any(term in csq for term in ["frameshift", "splice_acceptor", "splice_donor", "splice_canonical"]):
            verdict = VariantVerdict(
                variant=var,
                tier="Likely Pathogenic",
                evidence_source="Rule-based (predicted loss-of-function truncation/canonical splice disruption)",
                calibrated_score=0.950,
                clinical_notes="Canonical splice-site disruption or frameshift truncation in established tumor suppressor.",
            )

        # 3B. SpliceAI Deep Splicing Safeguard (Novel deep-intronic / synonymous splice disruptors)
        if verdict is None and var.spliceai_max >= 0.50:
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
        elif verdict is None and var.spliceai_max >= 0.20:
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

        # 3C. Extended Splice Region Boundary Warning (Unclassified non-canonical splice variant without SpliceAI)
        if verdict is None and "splice_region" in csq and not var.ref_aa and not var.alt_aa:
            verdict = VariantVerdict(
                variant=var,
                tier="VUS",
                evidence_source="Splice region boundary (unclassified non-canonical splice variant)",
                calibrated_score=0.400,
                clinical_notes="Located in non-canonical splice region (-3 to +8 bp / polypyrimidine tract). Requires RNA sequencing or SpliceAI annotation to confirm splicing impact.",
            )

        # 4. Deep Multimodal Model Inference for SNVs
        if verdict is None:
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

        if resolve_recessive:
            self._attach_risk_estimates(verdict)
        return verdict

    @staticmethod
    def _extract_phased_hap_slot(genotype: str) -> Optional[int]:
        """Returns the single phased alt haplotype index (0 or 1) if phased het, else None."""
        if "|" not in genotype:
            return None
        tokens = genotype.split("|")
        alt_slots = [idx for idx, tok in enumerate(tokens) if tok.isdigit() and int(tok) > 0]
        return alt_slots[0] if len(alt_slots) == 1 else None

    @classmethod
    def _resolve_compound_het_state(cls, het_vars: List[VariantRecord]) -> str:
        """Determines compound-het status ('single', 'trans', 'cis', or 'unphased') in a recessive gene."""
        unique_vars: Dict[tuple, VariantRecord] = {}
        for v in het_vars:
            unique_vars[(v.chrom, v.pos, v.ref, v.alt)] = v
        distinct = list(unique_vars.values())
        if len(distinct) < 2:
            return "single"

        has_unphased = False
        has_cis = False
        for i in range(len(distinct)):
            for j in range(i + 1, len(distinct)):
                va, vb = distinct[i], distinct[j]
                if va.chrom == vb.chrom and va.pos == vb.pos and va.alt != vb.alt:
                    # Two distinct alt alleles called at the same diploid locus (e.g. 1/2) are in trans
                    if any(len([t for t in re.split(r"[/|]", g) if t.isdigit() and int(t) > 0]) >= 2 for g in (va.genotype, vb.genotype)):
                        return "trans"
                ha = cls._extract_phased_hap_slot(va.genotype)
                hb = cls._extract_phased_hap_slot(vb.genotype)
                same_block = (
                    ha is not None
                    and hb is not None
                    and (
                        (va.phase_set is not None and va.phase_set == vb.phase_set)
                        or (va.phase_set is None and vb.phase_set is None)
                    )
                )
                if same_block:
                    if ha != hb:
                        return "trans"
                    has_cis = True
                else:
                    has_unphased = True

        if has_unphased:
            return "unphased"
        return "cis" if has_cis else "unphased"

    def evaluate_variants(self, variants: List[VariantRecord]) -> List[VariantVerdict]:
        """Evaluates a batch of variants from one sample, resolving recessive compound heterozygosity."""
        verdicts = [self._evaluate_variant_core(v, resolve_recessive=False) for v in variants]

        recessive_candidates: Dict[str, List[int]] = {}
        for idx, verdict in enumerate(verdicts):
            gene = verdict.variant.gene
            if (
                gene in RECESSIVE_GENES
                and verdict.tier in ("Pathogenic", "Likely Pathogenic")
                and not verdict.variant.is_biallelic
            ):
                recessive_candidates.setdefault(gene, []).append(idx)

        gene_comp_het_state: Dict[str, str] = {}
        for gene, idxs in recessive_candidates.items():
            gene_comp_het_state[gene] = self._resolve_compound_het_state(
                [verdicts[i].variant for i in idxs]
            )

        for idx, verdict in enumerate(verdicts):
            gene = verdict.variant.gene
            if gene in RECESSIVE_GENES and not verdict.variant.is_biallelic and verdict.tier in ("Pathogenic", "Likely Pathogenic"):
                state = gene_comp_het_state.get(gene, "single")
                if state == "trans":
                    verdict.clinical_notes += (
                        f" Compound heterozygous (phased in trans) in autosomal recessive gene {gene}; "
                        "biallelic criteria satisfied."
                    )
                    self._attach_risk_estimates(verdict, biallelic_confirmed=True)
                elif state == "unphased":
                    verdict.clinical_notes += (
                        f" Putative compound heterozygous in autosomal recessive gene {gene} "
                        "(two distinct heterozygous pathogenic/likely pathogenic variants detected without phasing; "
                        "parental or long-read phasing confirmation recommended to confirm trans configuration)."
                    )
                    self._attach_risk_estimates(verdict, biallelic_confirmed=True)
                elif state == "cis":
                    verdict.clinical_notes += (
                        f" Phased on the same haplotype (in cis) in {gene}; does not establish biallelic inactivation."
                    )
                    self._attach_risk_estimates(verdict, biallelic_confirmed=False)
                else:
                    self._attach_risk_estimates(verdict, biallelic_confirmed=False)
            else:
                self._attach_risk_estimates(verdict)

        return verdicts

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

        # Continuous biochemical features (exact parity with build_v3_features.py)
        has_prot = 1 if vclass in ["synonymous", "missense", "nonsense", "start_lost", "stop_lost"] else 0
        rel_pos = min(1.0, max(0.0, float(var.aa_pos) / float(prot_len))) if prot_len > 0 and var.aa_pos > 0 else 0.0

        from rdkit.Chem import DataStructs

        ref_1 = var.ref_aa
        alt_1 = var.alt_aa

        if vclass == "start_lost":
            rel_pos = 0.0
            grantham = 215.0
            tanimoto = 1.0
            dmw = -131.2
            dcharge = 0.0
            dlogp = 0.0
            dtpsa = 0.0
        elif vclass == "nonsense":
            grantham = 200.0
            tanimoto = 1.0
            if ref_1 in AA_PROPS:
                dmw = -AA_PROPS[ref_1]["mw"]
                dcharge = -AA_PROPS[ref_1]["charge"]
                dlogp = -AA_PROPS[ref_1]["logp"]
                dtpsa = -AA_PROPS[ref_1]["tpsa"]
            else:
                dmw, dcharge, dlogp, dtpsa = 0.0, 0.0, 0.0, 0.0
        elif vclass == "stop_lost":
            grantham = 200.0
            tanimoto = 1.0
            if alt_1 in AA_PROPS:
                dmw = AA_PROPS[alt_1]["mw"]
                dcharge = AA_PROPS[alt_1]["charge"]
                dlogp = AA_PROPS[alt_1]["logp"]
                dtpsa = AA_PROPS[alt_1]["tpsa"]
            else:
                dmw, dcharge, dlogp, dtpsa = 0.0, 0.0, 0.0, 0.0
        elif ref_1 in AA_PROPS and alt_1 in AA_PROPS:
            grantham = float(GRANTHAM.get((ref_1, alt_1), 100.0))
            p_ref, p_alt = AA_PROPS[ref_1], AA_PROPS[alt_1]
            dmw = round(p_alt["mw"] - p_ref["mw"], 3)
            dcharge = round(p_alt["charge"] - p_ref["charge"], 3)
            dlogp = round(p_alt["logp"] - p_ref["logp"], 3)
            dtpsa = round(p_alt["tpsa"] - p_ref["tpsa"], 3)
            tanimoto = 0.0 if ref_1 == alt_1 else round(1.0 - float(DataStructs.TanimotoSimilarity(p_ref["fp"], p_alt["fp"])), 4)
        else:
            grantham, tanimoto, dmw, dcharge, dlogp, dtpsa = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

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

            if self.ensemble_models:
                logits = [m(t_tab, t_vc, t_sy, t_llr, t_avail, t_emb).item() for m in self.ensemble_models]
                raw_logit = float(np.mean(logits))
            else:
                raw_logit = self.model(t_tab, t_vc, t_sy, t_llr, t_avail, t_emb).item()

            calibrated_logit = self.calibration_scale * raw_logit + self.calibration_bias
            calibrated_prob = float(1.0 / (1.0 + np.exp(-calibrated_logit)))

        return calibrated_prob

    def _attach_risk_estimates(self, verdict: VariantVerdict, biallelic_confirmed: bool = False):
        """Attaches published penetrance ranges and handles recessive zygosity."""
        gene = verdict.variant.gene

        # Only pathogenic or likely pathogenic variants convey carrier or disease risk
        if verdict.tier not in ["Pathogenic", "Likely Pathogenic"]:
            return

        # Handle recessive zygosity (MUTYH, NTHL1)
        if gene in RECESSIVE_GENES:
            if not verdict.variant.is_biallelic and not biallelic_confirmed:
                verdict.tier = "Carrier (Recessive)"
                if "in cis" not in verdict.clinical_notes:
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

        # Aggregate cumulative cancer risks by organ / cancer type, keeping each source range coupled to its gene
        cancer_risk_map = {}
        for v in actionable_verdicts:
            gene = v.variant.gene
            for r in v.cancer_risks:
                outcome = r["outcome"]
                if outcome not in cancer_risk_map:
                    cancer_risk_map[outcome] = {
                        "genes": set(),
                        "entries": [],
                    }
                cancer_risk_map[outcome]["genes"].add(gene)
                cancer_risk_map[outcome]["entries"].append({
                    "gene": gene,
                    "low": float(r["risk_low_pct"]),
                    "high": float(r["risk_high_pct"]),
                    "up_to_age": int(r["up_to_age"]),
                    "source": str(r["source"]),
                })

        # Top-level executive risk table
        cancer_summary_list = []
        if cancer_risk_map:
            lines.append("")
            lines.append("=" * 80)
            lines.append("EXECUTIVE SUMMARY: PREDICTED LIFETIME CANCER RISKS BY ORGAN / TYPE")
            lines.append("=" * 80)
            lines.append(f"{'CANCER TYPE':<32} | {'ESTIMATED LIFETIME RISK':<23} | {'GENE(S) INVOLVED':<16} | {'HORIZON'}")
            lines.append("-" * 80)
            
            # Sort by dominant estimate's upper bound risk descending
            def _dominant_entry(entries):
                return max(entries, key=lambda e: (e["high"], e["low"]))

            sorted_outcomes = sorted(
                cancer_risk_map.items(),
                key=lambda x: (_dominant_entry(x[1]["entries"])["high"], _dominant_entry(x[1]["entries"])["low"]),
                reverse=True,
            )
            for outcome, data in sorted_outcomes:
                genes_str = ", ".join(sorted(data["genes"]))
                dominant = _dominant_entry(data["entries"])
                risk_str = f"{dominant['low']:.0f}% - {dominant['high']:.0f}%"
                age_val = dominant["up_to_age"]
                age_str = f"Up to age {age_val}"
                lines.append(f"{outcome:<32} | {risk_str:<23} | {genes_str:<16} | {age_str}")
                cancer_summary_list.append({
                    "cancer_type": outcome,
                    "lifetime_risk_range": risk_str,
                    "genes": sorted(list(data["genes"])),
                    "up_to_age": age_val,
                })
            lines.append("-" * 80)
            lines.append("")

        for i, v in enumerate(verdicts, 1):
            var = v.variant
            lines.append(f"\nVARIANT #{i}: {var.gene} chr{var.chrom}:{var.pos} {var.ref}>{var.alt}")
            lines.append(f"  HGVS / Consequence: {var.hgvs_p or var.hgvs_c or var.consequence}")
            dp_str = str(var.dp) if var.dp is not None else "N/A"
            gq_str = str(var.gq) if var.gq is not None else "N/A"
            lines.append(f"  Genotype:           {var.genotype} (DP={dp_str}, GQ={gq_str})")
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
                        "id": v.variant.id,
                        "ref": v.variant.ref,
                        "alt": v.variant.alt,
                        "hgvs_c": v.variant.hgvs_c,
                        "hgvs_p": v.variant.hgvs_p,
                        "consequence": v.variant.consequence,
                        "genotype": v.variant.genotype,
                        "dp": v.variant.dp,
                        "gq": v.variant.gq,
                        "af": v.variant.af,
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
    parser.add_argument("--out_report", type=Path, default=PROJECT_ROOT / "reports" / "vcf_report.txt")
    parser.add_argument("--out_json", type=Path, default=PROJECT_ROOT / "reports" / "vcf_report.json")
    parser.add_argument("--out_pdf", type=Path, default=PROJECT_ROOT / "reports" / "vcf_report.pdf")
    parser.add_argument("--no_pdf", action="store_true", help="Disable automatic PDF generation")
    parser.add_argument("--min_dp", type=int, default=10)
    parser.add_argument("--min_gq", type=int, default=20)
    parser.add_argument("--device", type=str, default=None, help="Inference device (cpu or cuda)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    pipeline = VCFPipeline(device=args.device)
    variants = pipeline.parse_vcf(args.vcf_path, min_dp=args.min_dp, min_gq=args.min_gq)
    verdicts = pipeline.evaluate_variants(variants)
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
