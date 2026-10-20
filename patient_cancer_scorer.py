#!/usr/bin/env python3
"""
==============================================================================
GenomeDL v2: Person-Level Unified Hereditary Cancer Risk Scoring Engine
==============================================================================
Transforms patient constitutional genetic data (VCF / variant panel) into an
overall, continuous Person-Level Cancer Risk Score (0 to 100) using the
calibrated GenomeDL v2 Deep Tabular Residual Network.

Key Scientific & Algorithmic Principles:
1. Dynamic Variant Featurization: Evaluates any variant on the fly (no novel variant drops).
2. Deep Learning Pathogenicity: Uses calibrated P(Pathogenic) from GenomeDLResNetV2.
3. Biological Inheritance Modeling: Respects Autosomal Dominant (AD) vs Recessive (AR)
   zygosity (0/1 carrier vs 1/1 / compound heterozygous biallelic).
4. Multi-Variant Risk Compounding: Non-linear probabilistic complement compounding:
   Score = S_base + (100 - S_base) * [1 - Prod(1 - I_i)]
5. Organ-Specific Projections: Maps pathway impacts to clinical lifetime organ risks.
6. Actionable Clinical Surveillance: Aligned with NCCN & ACMG management guidelines.
"""

import os
import sys
import gzip
import re
import argparse
import warnings
warnings.filterwarnings('ignore')

# Force UTF-8 on Windows console
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import numpy as np
import pandas as pd

from train_v2_model import GenomeDLResNetV2, FEATURE_COLS, SCALE_COLS
from build_features_v2 import build_pairwise_biophysical_lookup, get_protein_lengths_dict, TRANSITIONS
from extract_clinvar_hereditary import PANEL, GENE2SYN, AA3

# ==============================================================================
# 1. CLINICAL GENETICS KNOWLEDGEBASE: INHERITANCE & PENETRANCE SPECIFICATIONS
# ==============================================================================

# Autosomal Recessive Genes (require biallelic hits for full disease penetrance)
AUTOSOMAL_RECESSIVE_GENES = {
    'MUTYH', 'NTHL1', 'BLM', 'WRN', 'RECQL4',
    'FANCA', 'FANCC', 'FANCD2', 'FANCE', 'FANCF', 'FANCG', 'FANCI', 'FANCL', 'FANCM', 'SLX4',
    'XPA', 'XPC', 'ERCC2', 'ERCC3', 'ERCC4', 'ERCC5', 'POLH', 'DDB2',
    'SBDS', 'CEP57', 'TRIP13'
}

# X-Linked Genes
X_LINKED_GENES = {'FANCB', 'DKC1'}

# High Penetrance Genes (Tier 1: w_g = 1.0)
HIGH_PENETRANCE_GENES = {
    'TP53', 'BRCA1', 'BRCA2', 'PALB2', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 'EPCAM',
    'APC', 'CDH1', 'PTEN', 'STK11', 'RB1', 'RET', 'VHL', 'WT1', 'CDKN2A', 'MEN1'
}

# Moderate Penetrance Genes (Tier 2: w_g = 0.65 - 0.75)
MODERATE_PENETRANCE_GENES = {
    'CHEK2', 'ATM', 'BARD1', 'BRIP1', 'RAD51C', 'RAD51D', 'NBN', 'NF1', 'NF2',
    'SDHA', 'SDHB', 'SDHC', 'SDHD', 'FH', 'FLCN', 'TSC1', 'TSC2',
    'POLE', 'POLD1', 'BAP1', 'SMAD4', 'BMPR1A'
}

def get_gene_penetrance_weight(gene):
    g = gene.upper()
    if g in HIGH_PENETRANCE_GENES:
        return 1.00
    elif g in AUTOSOMAL_RECESSIVE_GENES:
        return 0.95  # When biallelic; carrier logic applies zygosity attenuation
    elif g in MODERATE_PENETRANCE_GENES:
        return 0.70
    else:
        return 0.50  # Modifiers / Lower penetrance genes

# Primary Organ Risks & Pathway Associations
ORGAN_PATHWAYS = {
    'Breast': {
        'baseline_risk': 12.5,
        'max_risk': 75.0,
        'genes': {'BRCA1', 'BRCA2', 'PALB2', 'CHEK2', 'ATM', 'BARD1', 'RAD51C', 'RAD51D', 'TP53', 'PTEN', 'CDH1', 'STK11', 'NBN', 'NF1'}
    },
    'Ovarian': {
        'baseline_risk': 1.3,
        'max_risk': 45.0,
        'genes': {'BRCA1', 'BRCA2', 'PALB2', 'BRIP1', 'RAD51C', 'RAD51D', 'MLH1', 'MSH2', 'MSH6', 'PMS2'}
    },
    'Colorectal': {
        'baseline_risk': 4.2,
        'max_risk': 80.0,
        'genes': {'MLH1', 'MSH2', 'MSH6', 'PMS2', 'EPCAM', 'APC', 'MUTYH', 'NTHL1', 'SMAD4', 'BMPR1A', 'STK11', 'PTEN', 'POLE', 'POLD1', 'AXIN2', 'CHEK2'}
    },
    'Prostate': {
        'baseline_risk': 11.5,
        'max_risk': 25.0,
        'genes': {'BRCA2', 'BRCA1', 'HOXB13', 'ATM', 'CHEK2', 'PALB2', 'MLH1', 'MSH2'}
    },
    'Pancreatic': {
        'baseline_risk': 1.6,
        'max_risk': 12.0,
        'genes': {'BRCA2', 'BRCA1', 'PALB2', 'CDKN2A', 'ATM', 'MLH1', 'MSH2', 'STK11'}
    },
    'Endometrial': {
        'baseline_risk': 3.1,
        'max_risk': 55.0,
        'genes': {'MLH1', 'MSH2', 'MSH6', 'PMS2', 'EPCAM', 'PTEN'}
    },
    'Gastric': {
        'baseline_risk': 0.9,
        'max_risk': 70.0,
        'genes': {'CDH1', 'CTNNA1', 'MLH1', 'MSH2', 'MSH6', 'APC', 'SMAD4', 'BMPR1A', 'STK11', 'TP53'}
    },
    'CNS / Sarcoma': {
        'baseline_risk': 1.0,
        'max_risk': 80.0,
        'genes': {'TP53', 'NF1', 'NF2', 'RB1', 'SUFU', 'PTCH1', 'SMARCB1', 'SMARCA4'}
    }
}

# Clinical Management Guidelines (NCCN / ACMG)
CLINICAL_MANAGEMENT = {
    'Breast_Ovarian_HR_repair': [
        "Annual Contrast-Enhanced Breast MRI starting at age 25.",
        "Annual Mammography with tomosynthesis starting at age 30.",
        "Discuss Risk-Reducing Salpingo-Oophorectomy (RRSO) between ages 35-40.",
        "Cascade germline genetic testing recommended for all 1st-degree relatives."
    ],
    'Lynch_MMR_polymerase': [
        "High-definition colonoscopy every 1 to 2 years starting at age 20-25.",
        "Annual transvaginal ultrasound and endometrial biopsy starting at age 30-35.",
        "Consider daily aspirin chemoprevention under physician supervision.",
        "Routine immunohistochemistry (IHC) / MSI testing for any resected polyps."
    ],
    'Li_Fraumeni': [
        "Annual Whole-Body MRI (WBMRI) starting in early childhood.",
        "Annual Brain MRI and Contrast Breast MRI starting at age 20.",
        "Avoid unnecessary diagnostic ionizing radiation (CT scans / X-rays) when MRI/US is available.",
        "Comprehensive clinical physical exam every 6 months."
    ],
    'Polyposis_hamartoma': [
        "Flexible sigmoidoscopy / colonoscopy every 1-2 years starting at age 10-15.",
        "Upper GI endoscopy every 1-3 years upon gastric/duodenal polyposis.",
        "Consider prophylactic colectomy if polyp burden precludes endoscopic control."
    ],
    'General_Population_Baseline': [
        "Routine population age-based cancer screenings (mammogram at age 40/45, colonoscopy at age 45).",
        "Maintain balanced lifestyle, regular physical activity, and avoid tobacco exposure.",
        "No high-penetrance constitutional predisposition alterations detected."
    ]
}

# ==============================================================================
# 2. VCF PARSING & DYNAMIC VARIANT PREPARATION
# ==============================================================================

# Regex for protein change parsing
PROT_P_RE = re.compile(r"p\.([A-Z][a-z]{2}|\*|[A-Z])(\d+)([A-Z][a-z]{2}|\*|[A-Z]|=|\?)")

def parse_protein_hgvs(hgvs_str):
    """
    Parses HGVS protein notation (e.g. p.Cys61Gly or p.C61G or p.Lys1192Lys).
    Returns (ref_aa, alt_aa, aa_pos, variant_class)
    """
    if not hgvs_str or hgvs_str == '.':
        return '.', '.', 0, 'intronic'
        
    m = PROT_P_RE.search(hgvs_str)
    if m:
        ref_raw, pos_str, alt_raw = m.groups()
        pos = int(pos_str)
        
        # Convert 3-letter or 1-letter
        ref = AA3.get(ref_raw, ref_raw if len(ref_raw) == 1 else '?')
        if alt_raw in ('=', ref_raw):
            alt = ref
            vclass = 'synonymous'
        elif alt_raw in ('Ter', '*'):
            alt = '*'
            vclass = 'nonsense'
        else:
            alt = AA3.get(alt_raw, alt_raw if len(alt_raw) == 1 else '?')
            if ref == 'M' and pos == 1 and alt != 'M':
                vclass = 'start_lost'
            else:
                vclass = 'missense'
        return ref, alt, pos, vclass
        
    return '.', '.', 0, 'intronic'

def parse_patient_vcf(vcf_path):
    """
    Parses a patient VCF file (.vcf or .vcf.gz) and yields structured variant records.
    """
    if not os.path.exists(vcf_path):
        raise FileNotFoundError(f"Patient VCF file not found at: {vcf_path}")
        
    records = []
    is_gz = vcf_path.endswith('.gz')
    open_fn = gzip.open if is_gz else open
    mode = 'rt' if is_gz else 'r'
    
    with open_fn(vcf_path, mode, encoding='utf-8', errors='ignore') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 5:
                continue
                
            chrom = parts[0].replace('chr', '').strip()
            pos = int(parts[1])
            vcf_id = parts[2]
            ref = parts[3].upper()
            alt = parts[4].upper()
            qual = parts[5] if len(parts) > 5 else '.'
            filt = parts[6] if len(parts) > 6 else '.'
            info = parts[7] if len(parts) > 7 else '.'
            
            # Extract Genotype (GT)
            gt = "0/1"
            if len(parts) >= 10:
                fmt = parts[8].split(':')
                sample = parts[9].split(':')
                if 'GT' in fmt:
                    gt = sample[fmt.index('GT')]
                    
            # Parse INFO fields (GENE, HGVSp, etc.)
            gene = None
            hgvsp = None
            for item in info.split(';'):
                if item.startswith('GENE='):
                    gene = item.split('=')[1].strip().upper()
                elif item.startswith('HGVSp='):
                    hgvsp = item.split('=')[1].strip()
                elif 'p.' in item and not hgvsp:
                    hgvsp = item.strip()
                    
            records.append({
                'chrom': chrom,
                'pos': pos,
                'vcf_id': vcf_id,
                'ref': ref,
                'alt': alt,
                'qual': qual,
                'filter': filt,
                'info': info,
                'genotype': gt,
                'gene': gene,
                'hgvsp': hgvsp
            })
            
    return records

# ==============================================================================
# 3. PERSON-LEVEL CANCER RISK SCORING ENGINE
# ==============================================================================

class PersonCancerScorer:
    """
    End-to-End Person-Level Hereditary Cancer Risk Scoring Engine.
    Combines calibrated Deep Learning variant pathogenicity inference with
    clinical inheritance genetics to compute an overall 0-100 Person Cancer Score.
    """
    def __init__(self, model_checkpoint_path=None, catalog_path=None):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        if model_checkpoint_path is None:
            model_checkpoint_path = os.path.join(script_dir, "models", "v2_pan_cancer_resnet.pth")
        if catalog_path is None:
            catalog_path = os.path.join(script_dir, "data", "v2_pan_cancer_featured_labelled.csv")
            
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Model Checkpoint
        if not os.path.exists(model_checkpoint_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_checkpoint_path}")
            
        ckpt = torch.load(model_checkpoint_path, map_location=self.device, weights_only=False)
        self.gene_to_id = ckpt['gene_to_id']
        self.syn_to_id = ckpt.get('syndrome_to_id', {})
        self.scaler = ckpt['scaler']
        self.temperature = float(ckpt.get('temperature', 0.8769))
        
        self.model = GenomeDLResNetV2(
            num_cont=len(FEATURE_COLS),
            num_genes=len(self.gene_to_id),
            num_syndromes=len(self.syn_to_id),
            hidden_dim=256
        ).to(self.device)
        self.model.load_state_dict(ckpt['model_state_dict'])
        self.model.eval()
        
        # Pre-compute biophysical lookups & protein lengths for on-the-fly scoring
        self.pairwise_lookup = build_pairwise_biophysical_lookup()
        out_dir = os.path.join(script_dir, "out")
        self.prot_lens = get_protein_lengths_dict(out_dir)
        
        # Build catalog lookup for fast annotation of known variants
        self.catalog_lookup = {}
        self.gene_coord_spans = {}
        if os.path.exists(catalog_path):
            cat_df = pd.read_csv(catalog_path, usecols=['gene', 'syndrome_group', 'chrom', 'pos', 'ref', 'alt', 'ref_aa', 'alt_aa', 'aa_pos', 'variant_class'])
            for _, r in cat_df.iterrows():
                key = (str(r['chrom']), int(r['pos']), str(r['ref']), str(r['alt']))
                if key not in self.catalog_lookup:
                    self.catalog_lookup[key] = {
                        'gene': str(r['gene']).upper(),
                        'syndrome_group': str(r['syndrome_group']),
                        'ref_aa': str(r['ref_aa']),
                        'alt_aa': str(r['alt_aa']),
                        'aa_pos': int(r['aa_pos']),
                        'variant_class': str(r['variant_class'])
                    }
                    
            # Compute gene coordinate intervals
            grouped = cat_df.groupby('gene')
            for g, grp in grouped:
                c = str(grp['chrom'].iloc[0])
                p_min = int(grp['pos'].min())
                p_max = int(grp['pos'].max())
                self.gene_coord_spans[g.upper()] = (c, p_min, p_max)

    def resolve_gene_for_locus(self, chrom, pos):
        """
        Maps a genomic coordinate to one of the 130 panel genes.
        """
        c = str(chrom).replace('chr', '')
        for gene, (g_chrom, p_min, p_max) in self.gene_coord_spans.items():
            if g_chrom == c and (p_min - 10000 <= pos <= p_max + 10000):
                return gene
        return None

    def featurize_variant_record(self, rec):
        """
        Featurizes a variant into the model's 29 continuous and one-hot features.
        Zero variants dropped: Checks pre-indexed catalog first; if novel, featurizes on the fly.
        """
        chrom = str(rec['chrom']).replace('chr', '')
        pos = int(rec['pos'])
        ref = str(rec['ref']).upper()
        alt = str(rec['alt']).upper()
        key = (chrom, pos, ref, alt)
        
        gene = rec.get('gene')
        if not gene or gene not in self.gene_to_id:
            resolved = self.resolve_gene_for_locus(chrom, pos)
            gene = resolved if resolved else (gene if gene else 'BRCA1')
            
        gene = gene.upper()
        syndrome = GENE2SYN.get(gene, 'Other_hereditary')
        
        # Case A: Known variant in pre-indexed catalog
        if key in self.catalog_lookup:
            cat = self.catalog_lookup[key]
            ref_aa = cat['ref_aa']
            alt_aa = cat['alt_aa']
            aa_pos = cat['aa_pos']
            vclass = cat['variant_class']
        else:
            # Case B: Novel variant - extract from HGVSp or compute
            hgvsp = rec.get('hgvsp')
            if hgvsp:
                ref_aa, alt_aa, aa_pos, vclass = parse_protein_hgvs(hgvsp)
            else:
                ref_aa = rec.get('ref_aa', '.')
                alt_aa = rec.get('alt_aa', '.')
                aa_pos = int(rec.get('aa_pos', 0))
                vclass = rec.get('variant_class', 'missense' if ref_aa != alt_aa and ref_aa != '.' else 'intronic')
                
        var_dict = {
            'gene': gene,
            'chrom': chrom,
            'pos': pos,
            'ref': ref,
            'alt': alt,
            'ref_aa': ref_aa,
            'alt_aa': alt_aa,
            'aa_pos': aa_pos,
            'variant_class': vclass
        }
        
        # Featurize using canonical biophysical pipeline
        # 1. Base one-hots
        ref_onehot = [1.0 if ref == b else 0.0 for b in ['A', 'C', 'G', 'T']]
        alt_onehot = [1.0 if alt == b else 0.0 for b in ['A', 'C', 'G', 'T']]
        is_trans = 1.0 if (ref, alt) in TRANSITIONS else 0.0
        
        # 2. Molecular class flags
        is_miss = 1.0 if vclass == 'missense' else 0.0
        is_non = 1.0 if vclass == 'nonsense' or alt_aa == '*' else 0.0
        is_syn = 1.0 if vclass == 'synonymous' or (ref_aa == alt_aa and ref_aa != '.') else 0.0
        is_sl = 1.0 if vclass == 'start_lost' or (ref_aa == 'M' and aa_pos == 1 and alt_aa != 'M') else 0.0
        is_sc = 1.0 if vclass == 'splice_canonical' else 0.0
        is_sr = 1.0 if vclass == 'splice_region' else 0.0
        is_int = 1.0 if vclass == 'intronic' else 0.0
        is_utr = 1.0 if vclass in ('utr5', 'utr3') else 0.0
        is_onc = 1.0 if vclass in ('other_noncoding', 'coding_no_protein_annot') else 0.0
        
        # 3. Position metrics
        g_len = self.prot_lens.get(gene, 1000)
        if is_miss or is_non or is_syn or is_sl:
            rel_pos = min(1.0, max(0.0, float(aa_pos) / float(g_len)))
            has_prot = 1.0
        else:
            rel_pos = 0.0
            has_prot = 0.0
            
        # 4. Biophysical features
        if (ref_aa, alt_aa) in self.pairwise_lookup:
            (g_dist, b_score, d_mw, d_logp, d_tpsa, d_chg, c_dist, pol_chg, chg_chg) = self.pairwise_lookup[(ref_aa, alt_aa)]
        elif is_non:
            g_dist, b_score, d_mw, d_logp, d_tpsa, d_chg, c_dist, pol_chg, chg_chg = (200.0, -4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0)
        elif is_syn:
            g_dist, b_score, d_mw, d_logp, d_tpsa, d_chg, c_dist, pol_chg, chg_chg = (0.0, 4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            g_dist, b_score, d_mw, d_logp, d_tpsa, d_chg, c_dist, pol_chg, chg_chg = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
            
        feat_vector = (
            ref_onehot + alt_onehot + [is_trans] +
            [is_miss, is_non, is_syn, is_sl, is_sc, is_sr, is_int, is_utr, is_onc] +
            [has_prot, rel_pos, g_dist, b_score, d_mw, d_logp, d_tpsa, d_chg, c_dist, pol_chg, chg_chg]
        )
        
        var_dict['feat_vector'] = np.array(feat_vector, dtype=np.float32)
        var_dict['syndrome'] = syndrome
        var_dict['grantham'] = g_dist
        var_dict['genotype'] = rec.get('genotype', '0/1')
        return var_dict

    def score_single_variant(self, var_dict):
        """
        Evaluates calibrated ResNet pathogenicity for a single featurized variant.
        """
        gene = var_dict['gene']
        gene_id = self.gene_to_id.get(gene, 0)
        syndrome = var_dict['syndrome']
        syn_id = self.syn_to_id.get(syndrome, 0)
        
        xc = var_dict['feat_vector'].copy()
        # Scale continuous features
        sub_idx = [FEATURE_COLS.index(c) for c in SCALE_COLS]
        xc[sub_idx] = self.scaler.transform([xc[sub_idx]])[0]
        
        gt = torch.tensor([gene_id], dtype=torch.long).to(self.device)
        st = torch.tensor([syn_id], dtype=torch.long).to(self.device)
        xt = torch.tensor(np.array([xc]), dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            logit = self.model(xt, gt, st).item()
            prob = 1.0 / (1.0 + np.exp(-logit / self.temperature))
            
        if prob >= 0.85:
            acmg_class = "CLASS 4: PATHOGENIC"
        elif prob >= 0.65:
            acmg_class = "CLASS 3: LIKELY PATHOGENIC"
        elif prob >= 0.35:
            acmg_class = "CLASS 2: VUS (UNCERTAIN)"
        elif prob >= 0.10:
            acmg_class = "CLASS 1: LIKELY BENIGN"
        else:
            acmg_class = "CLASS 0: BENIGN"
            
        return logit, prob, acmg_class

    def score_patient(self, patient_id, variant_records, base_cancer_risk=12.0):
        """
        Main Clinical Pipeline: Ingests all variants across a patient's panel,
        runs Deep Learning inference, applies inheritance / zygosity weighting,
        and computes the Unified Person-Level Cancer Risk Score (0 to 100).
        """
        if not variant_records:
            return self._build_baseline_result(patient_id, base_cancer_risk)
            
        # 1. Featurize and evaluate every variant
        evaluated_variants = []
        for rec in variant_records:
            var_dict = self.featurize_variant_record(rec)
            logit, prob, acmg = self.score_single_variant(var_dict)
            var_dict['logit'] = logit
            var_dict['prob'] = prob
            var_dict['acmg_class'] = acmg
            evaluated_variants.append(var_dict)
            
        # 2. Check for compound heterozygosity in Autosomal Recessive genes
        ar_damaging_counts = {}
        for v in evaluated_variants:
            g = v['gene']
            if g in AUTOSOMAL_RECESSIVE_GENES and v['prob'] >= 0.35:
                ar_damaging_counts[g] = ar_damaging_counts.get(g, 0) + 1
                
        # 3. Calculate effective pathogenic impact (I_i) per variant
        for v in evaluated_variants:
            g = v['gene']
            p = v['prob']
            gt = v['genotype']
            w_gene = get_gene_penetrance_weight(g)
            
            # Excess damaging probability above benign baseline (0.15)
            # Variants with p <= 0.15 contribute exactly 0 excess risk
            excess_p = max(0.0, (p - 0.15) / (1.0 - 0.15))
            
            # Zygosity factor (z_i)
            if g in AUTOSOMAL_RECESSIVE_GENES:
                if '1/1' in gt or ar_damaging_counts.get(g, 0) >= 2:
                    z = 1.0  # Biallelic (homozygous 1/1 or compound heterozygous)
                    status_str = "BIALLELIC (RECESSIVE DISEASE)"
                else:
                    z = 0.10 # Monoallelic carrier (mild / modifier only)
                    status_str = "MONOALLELIC CARRIER"
            else:
                z = 1.0      # Autosomal Dominant / X-linked
                status_str = "HETEROZYGOUS" if '0/1' in gt else "HOMOZYGOUS"
                
            effective_impact = z * w_gene * excess_p
            v['excess_p'] = excess_p
            v['zygosity_factor'] = z
            v['gene_weight'] = w_gene
            v['effective_impact'] = effective_impact
            v['inheritance_status'] = status_str
            
        # 4. Multi-Variant Multiplicative Compounding for Overall Person Score
        # Bounded complement: P_composite = 1 - Prod(1 - I_i)
        prod_complement = 1.0
        for v in evaluated_variants:
            prod_complement *= (1.0 - v['effective_impact'])
        p_composite_excess = 1.0 - prod_complement
        
        person_cancer_score = base_cancer_risk + (100.0 - base_cancer_risk) * p_composite_excess
        person_cancer_score = round(min(100.0, max(base_cancer_risk, person_cancer_score)), 1)
        
        # 5. Organ-Specific Risk Radar
        organ_projections = {}
        for organ, o_info in ORGAN_PATHWAYS.items():
            base_r = o_info['baseline_risk']
            max_r = o_info['max_risk']
            o_genes = o_info['genes']
            
            # Compounding for organ-specific pathway genes
            o_prod = 1.0
            for v in evaluated_variants:
                if v['gene'] in o_genes:
                    o_prod *= (1.0 - v['effective_impact'])
            o_excess = 1.0 - o_prod
            projected_organ_risk = base_r + (max_r - base_r) * o_excess
            organ_projections[organ] = {
                'projected_risk': round(projected_organ_risk, 1),
                'baseline_risk': base_r,
                'delta': round(projected_organ_risk - base_r, 1),
                'is_elevated': projected_organ_risk > (base_r + 2.0)
            }
            
        # 6. Sort variants by impact
        evaluated_variants.sort(key=lambda x: x['effective_impact'], reverse=True)
        top_driver = evaluated_variants[0] if evaluated_variants else None
        
        # 7. Clinical Risk Tier Mapping
        if person_cancer_score >= 75.0:
            risk_tier = "HIGH HEREDITARY CANCER RISK"
            tier_badge = "[ CRITICAL HIGH RISK ]"
            clinical_code = "HIGH_RISK"
        elif person_cancer_score >= 50.0:
            risk_tier = "ELEVATED HEREDITARY CANCER RISK"
            tier_badge = "[ ELEVATED RISK ]"
            clinical_code = "ELEVATED_RISK"
        elif person_cancer_score >= 25.0:
            risk_tier = "MODERATE RISK / VUS"
            tier_badge = "[ MODERATE RISK ]"
            clinical_code = "MODERATE_RISK"
        else:
            risk_tier = "LOW / POPULATION BASELINE RISK"
            tier_badge = "[ POPULATION BASELINE ]"
            clinical_code = "BASELINE_RISK"
            
        # Clinical surveillance guidelines
        primary_syndrome = top_driver['syndrome'] if (top_driver and top_driver['effective_impact'] > 0.1) else 'General_Population_Baseline'
        surveillance_plan = CLINICAL_MANAGEMENT.get(primary_syndrome, CLINICAL_MANAGEMENT['General_Population_Baseline'])
        
        return {
            'patient_id': patient_id,
            'person_cancer_score': person_cancer_score,
            'risk_tier': risk_tier,
            'tier_badge': tier_badge,
            'clinical_code': clinical_code,
            'base_risk': base_cancer_risk,
            'p_composite_excess': p_composite_excess,
            'primary_syndrome': primary_syndrome,
            'top_driver': top_driver,
            'screened_variants': evaluated_variants,
            'organ_projections': organ_projections,
            'surveillance_plan': surveillance_plan
        }

    def _build_baseline_result(self, patient_id, base_risk):
        organ_projections = {}
        for organ, o_info in ORGAN_PATHWAYS.items():
            base_r = o_info['baseline_risk']
            organ_projections[organ] = {
                'projected_risk': base_r,
                'baseline_risk': base_r,
                'delta': 0.0,
                'is_elevated': False
            }
        return {
            'patient_id': patient_id,
            'person_cancer_score': base_risk,
            'risk_tier': "LOW / POPULATION BASELINE RISK",
            'tier_badge': "[ POPULATION BASELINE ]",
            'clinical_code': "BASELINE_RISK",
            'base_risk': base_risk,
            'p_composite_excess': 0.0,
            'primary_syndrome': 'General_Population_Baseline',
            'top_driver': None,
            'screened_variants': [],
            'organ_projections': organ_projections,
            'surveillance_plan': CLINICAL_MANAGEMENT['General_Population_Baseline']
        }

# ==============================================================================
# 4. CLINICAL TERMINAL & HTML REPORT GENERATORS
# ==============================================================================

def print_person_cancer_report(report):
    """
    Renders an ASCII clinical genetics report with unified score and breakdown.
    """
    p_id = report['patient_id']
    score = report['person_cancer_score']
    tier = report['risk_tier']
    badge = report['tier_badge']
    variants = report['screened_variants']
    organs = report['organ_projections']
    surv = report['surveillance_plan']
    
    # Progress visual bar (width 30)
    filled = int(round((score / 100.0) * 30))
    bar = "#" * filled + "-" * (30 - filled)
    
    print("\n" + "="*84)
    print("      GENOMEDL V2: PERSON-LEVEL HEREDITARY CANCER RISK SCORE REPORT")
    print(f"      Patient ID: {p_id} | Reference Build: GRCh38 | Panel: 130 Predisposition Genes")
    print("="*84)
    
    print("\n[★] UNIFIED PERSON-LEVEL HEREDITARY CANCER SCORE:")
    print("    " + "-"*58)
    print(f"    >>> SCORE: {score:4.1f} / 100  {badge} <<<")
    print(f"    Visual Scale: [{bar}] ({score:.1f}%)")
    print(f"    Clinical Tier: {tier} (Population Baseline: {report['base_risk']:.1f}/100)")
    print("    " + "-"*58)
    
    if report['top_driver'] and report['top_driver']['effective_impact'] > 0.05:
        td = report['top_driver']
        p_change = f"p.{td['ref_aa']}{td['aa_pos']}{td['alt_aa']}" if td['ref_aa'] != '.' else td['variant_class']
        print(f"\n[!] PRIMARY PREDISPOSITION DRIVER:")
        print(f"    * Gene Alteration: {td['gene']} {p_change} (chr{td['chrom']}:{td['pos']} {td['ref']}>{td['alt']})")
        print(f"    * ResNet P(Damaging): {td['prob']*100:.2f}% | ACMG: {td['acmg_class']}")
        print(f"    * Status: {td['inheritance_status']} | Gene Penetrance: {td['gene_weight']:.2f}")
    else:
        print(f"\n[OK] CLINICAL FINDINGS: No high-penetrance damaging variants detected across 130 genes.")
        
    print(f"\n[1] SCREENED PANEL VARIANTS BREAKDOWN ({len(variants)} Variants Analyzed):")
    print(f"    {'Gene':<8}{'Locus':<18}{'Variant':<10}{'Protein Change':<16}{'P(Damaging)':<14}{'Impact (I_i)':<14}{'Status'}")
    print("    " + "-"*86)
    for v in variants:
        p_str = f"p.{v['ref_aa']}{v['aa_pos']}{v['alt_aa']}" if v['ref_aa'] != '.' else v['variant_class']
        locus = f"chr{v['chrom']}:{v['pos']}"
        mut = f"{v['ref']}>{v['alt']}"
        print(f"    {v['gene']:<8}{locus:<18}{mut:<10}{p_str:<16}{v['prob']*100:>6.2f}%       {v['effective_impact']:>6.3f}        {v['inheritance_status']}")
    print("    " + "-"*86)
    
    print(f"\n[2] ORGAN-SPECIFIC LIFETIME CANCER RISK RADAR:")
    for organ, o_data in organs.items():
        curr_r = o_data['projected_risk']
        base_r = o_data['baseline_risk']
        delta_str = f"({o_data['delta']:+.1f}% vs baseline {base_r:.1f}%)" if o_data['is_elevated'] else f"(Baseline: {base_r:.1f}%)"
        flag = "[!]" if o_data['is_elevated'] else "[ ]"
        
        o_bar_len = int(min(20, round((curr_r / 100.0) * 20)))
        o_bar = "#" * o_bar_len + "-" * (20 - o_bar_len)
        print(f"    {flag} {organ:<16} : {curr_r:>5.1f}% [{o_bar}] {delta_str}")
        
    print(f"\n[3] ACTIONABLE CLINICAL SURVEILLANCE & PREVENTION (NCCN GUIDELINES):")
    for protocol in surv:
        print(f"    * {protocol}")
    print("="*84 + "\n")

def export_html_report(report, output_path):
    """
    Exports a styled HTML genetic report.
    """
    p_id = report['patient_id']
    score = report['person_cancer_score']
    tier = report['risk_tier']
    variants = report['screened_variants']
    organs = report['organ_projections']
    surv = report['surveillance_plan']
    
    if score >= 75:
        theme_color = "#dc2626"
        badge_bg = "#fee2e2"
        badge_border = "#f87171"
    elif score >= 50:
        theme_color = "#ea580c"
        badge_bg = "#ffedd5"
        badge_border = "#fb923c"
    elif score >= 25:
        theme_color = "#d97706"
        badge_bg = "#fef3c7"
        badge_border = "#fcd34d"
    else:
        theme_color = "#16a34a"
        badge_bg = "#dcfce7"
        badge_border = "#86efac"
        
    rows = ""
    for v in variants:
        p_str = f"p.{v['ref_aa']}{v['aa_pos']}{v['alt_aa']}" if v['ref_aa'] != '.' else v['variant_class']
        rows += f"""
        <tr>
            <td><strong>{v['gene']}</strong></td>
            <td>chr{v['chrom']}:{v['pos']}</td>
            <td>{v['ref']} &gt; {v['alt']}</td>
            <td><code>{p_str}</code></td>
            <td>{v['genotype']}</td>
            <td><strong>{v['prob']*100:.2f}%</strong></td>
            <td>{v['acmg_class']}</td>
            <td>{v['effective_impact']:.3f}</td>
        </tr>
        """
        
    organ_rows = ""
    for organ, o_data in organs.items():
        flag_badge = f"<span style='color:{theme_color};font-weight:bold;'>ELEVATED</span>" if o_data['is_elevated'] else "<span style='color:#6b7280;'>BASELINE</span>"
        organ_rows += f"""
        <tr>
            <td><strong>{organ}</strong></td>
            <td><strong>{o_data['projected_risk']:.1f}%</strong></td>
            <td>{o_data['baseline_risk']:.1f}%</td>
            <td>{flag_badge}</td>
        </tr>
        """
        
    surv_items = "".join([f"<li>{s}</li>" for s in surv])
    
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Hereditary Cancer Risk Report - {p_id}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; padding: 24px; }}
        .container {{ max-width: 900px; margin: 0 auto; background: #ffffff; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); padding: 32px; border: 1px solid #e2e8f0; }}
        .header {{ border-bottom: 2px solid #e2e8f0; padding-bottom: 16px; margin-bottom: 24px; }}
        .score-card {{ background: {badge_bg}; border: 2px solid {badge_border}; border-radius: 12px; padding: 24px; text-align: center; margin-bottom: 28px; }}
        .score-val {{ font-size: 56px; font-weight: 800; color: {theme_color}; margin: 8px 0; }}
        .tier-label {{ font-size: 20px; font-weight: 700; color: {theme_color}; text-transform: uppercase; }}
        table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
        th, td {{ padding: 10px 12px; border-bottom: 1px solid #e2e8f0; text-align: left; font-size: 14px; }}
        th {{ background: #f1f5f9; font-weight: 600; color: #475569; }}
        .bar-outer {{ background: #e2e8f0; border-radius: 999px; height: 16px; width: 100%; margin-top: 12px; overflow: hidden; }}
        .bar-inner {{ background: {theme_color}; height: 100%; border-radius: 999px; width: {score}%; }}
        ul {{ line-height: 1.6; color: #334155; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="margin:0 0 8px 0; font-size: 24px;">GenomeDL v2: Hereditary Cancer Risk Report</h1>
            <p style="margin:0; color:#64748b; font-size: 14px;">Patient ID: <strong>{p_id}</strong> | Reference: GRCh38 | Panel: 130 Predisposition Genes</p>
        </div>
        
        <div class="score-card">
            <div style="font-size: 14px; font-weight: 600; color: #475569; letter-spacing: 0.05em; text-transform: uppercase;">Overall Hereditary Cancer Risk Score</div>
            <div class="score-val">{score:.1f} / 100</div>
            <div class="tier-label">{tier}</div>
            <div class="bar-outer"><div class="bar-inner"></div></div>
            <p style="margin: 12px 0 0 0; font-size: 13px; color: #64748b;">General Population Lifetime Cancer Risk Baseline: 12.0 / 100</p>
        </div>
        
        <h3>1. Screened Genetic Variants Summary</h3>
        <table>
            <thead>
                <tr>
                    <th>Gene</th><th>Locus</th><th>Variant</th><th>Protein Change</th><th>Genotype</th><th>P(Damaging)</th><th>ACMG Class</th><th>Impact (I_i)</th>
                </tr>
            </thead>
            <tbody>
                {rows}
            </tbody>
        </table>
        
        <h3>2. Organ-Specific Lifetime Projections</h3>
        <table>
            <thead>
                <tr>
                    <th>Target Organ</th><th>Projected Risk</th><th>Population Baseline</th><th>Clinical Flag</th>
                </tr>
            </thead>
            <tbody>
                {organ_rows}
            </tbody>
        </table>
        
        <h3>3. Actionable Clinical Surveillance Protocols (NCCN Guidelines)</h3>
        <ul>
            {surv_items}
        </ul>
    </div>
</body>
</html>
"""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)
    print(f"[+] HTML clinical report exported to: {output_path}")

# ==============================================================================
# 5. CLI INTERFACE & DEMO BENCHMARKS
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="GenomeDL v2 Person-Level Cancer Risk Scoring Engine")
    parser.add_argument("--vcf", type=str, default=None, help="Path to patient VCF file (.vcf or .vcf.gz)")
    parser.add_argument("--patient_id", type=str, default="PATIENT_SAMPLE", help="Patient identifier")
    parser.add_argument("--html", type=str, default=None, help="Optional output path for HTML report")
    parser.add_argument("--demo", action="store_true", help="Run standard clinical demo cases (Sarah, Control, Lynch)")
    parser.add_argument("--demo_all", action="store_true", help="Run all 6 benchmark cohorts")
    args = parser.parse_args()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    scorer = PersonCancerScorer()
    
    if args.demo or args.demo_all:
        cases = [
            ("DEMO_PATIENT_SARAH_24Y", os.path.join(script_dir, "demo_vcfs", "patient_sarah_hboc.vcf")),
            ("DEMO_PATIENT_CONTROL_30Y", os.path.join(script_dir, "demo_vcfs", "patient_healthy_control.vcf")),
            ("DEMO_PATIENT_LYNCH_40Y", os.path.join(script_dir, "demo_vcfs", "patient_lynch_carrier.vcf")),
        ]
        if args.demo_all:
            cases += [
                ("DEMO_PATIENT_COMPOUND_MODERATE", os.path.join(script_dir, "demo_vcfs", "patient_compound_moderate.vcf")),
                ("DEMO_PATIENT_MUTYH_CARRIER", os.path.join(script_dir, "demo_vcfs", "patient_recessive_carrier.vcf")),
                ("DEMO_PATIENT_MUTYH_BIALLELIC", os.path.join(script_dir, "demo_vcfs", "patient_recessive_biallelic.vcf")),
                ("DEMO_PATIENT_NOVEL_TP53", os.path.join(script_dir, "demo_vcfs", "patient_novel_variant.vcf")),
            ]
            
        for p_id, vcf_p in cases:
            if os.path.exists(vcf_p):
                records = parse_patient_vcf(vcf_p)
                report = scorer.score_patient(p_id, records)
                print_person_cancer_report(report)
            else:
                print(f"[!] Warning: Demo VCF not found at {vcf_p}")
        return
        
    if args.vcf:
        records = parse_patient_vcf(args.vcf)
        report = scorer.score_patient(args.patient_id, records)
        print_person_cancer_report(report)
        if args.html:
            export_html_report(report, args.html)
    else:
        # Default run: Sarah HBOC case
        sarah_vcf = os.path.join(script_dir, "demo_vcfs", "patient_sarah_hboc.vcf")
        if os.path.exists(sarah_vcf):
            records = parse_patient_vcf(sarah_vcf)
            report = scorer.score_patient("DEMO_PATIENT_SARAH_24Y", records)
            print_person_cancer_report(report)
            default_html = os.path.join(script_dir, "demo_vcfs", "sarah_hboc_report.html")
            export_html_report(report, default_html)
        else:
            print("[!] Please provide a VCF file via --vcf <path> or use --demo")

if __name__ == '__main__':
    main()
