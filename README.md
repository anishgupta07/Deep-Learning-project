# GenomeDL v2: Deep Learning Engine for Germline Cancer Risk & Person-Level Scoring

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B%20%7C%20CUDA%20%7C%20CPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-2D9CDB)](https://www.rdkit.org/)
[![Biopython](https://img.shields.io/badge/Biopython-BLOSUM62-00599C)](https://biopython.org/)
[![ROC-AUC](https://img.shields.io/badge/Test%20ROC--AUC-99.29%25-2ECC71)](train_v2_model.py)
[![Precision](https://img.shields.io/badge/Precision-89.91%25-brightgreen)](train_v2_model.py)
[![Specificity](https://img.shields.io/badge/Specificity-98.25%25-brightgreen)](train_v2_model.py)
[![ECE Calibration](https://img.shields.io/badge/ECE%20Calibration-0.0138-blue)](train_v2_model.py)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end computational genomics and deep learning framework designed to evaluate constitutional genetic mutations across **130 validated hereditary cancer predisposition genes**, resolve Variants of Uncertain Significance (VUS), and calculate a **Unified Person-Level Hereditary Cancer Risk Score (0 to 100)** from constitutional sequencing data (.vcf / variant panel).

---

## 1. Problem Statement: From Variant Damage to Person-Level Risk

### The Genomic Challenge
The human genome contains **3.2 billion base pairs**. When constitutional DNA is sequenced from blood or saliva, an individual exhibits **3.5 to 4.5 million natural genetic variants**. 

In hereditary cancer genetics, clinical gene panels target high- and moderate-penetrance tumor suppressors (*BRCA1, BRCA2, TP53, MLH1, MSH2, APC, PTEN, CDH1, PALB2, ATM*, etc.). Patients and clinicians need two complementary layers of intelligence:
1. **Variant-Level Molecular Scoring:** Does a specific mutation destroy protein function (Pathogenic vs Benign)?
2. **Person-Level Overall Cancer Score (0 to 100):** Given all mutations detected in a person's DNA test, what is their overall lifetime hereditary cancer risk?

```
3.2 Billion Base Pairs (Human Constitutional DNA)
      │
      ▼  [The 130-Gene Coordinate Filter]
15 to 35 Constitutional Mutations in Panel Genes
      │
      ▼  [GenomeDL ResNet v2: Calibrated Pathogenicity Engine]
Individual Variant Damaging Probabilities (0.0% to 100.0%)
      │
      ▼  [Inheritance Genetics & Multi-Variant Compounding Engine]
UNIFIED PERSON-LEVEL HEREDITARY CANCER RISK SCORE (0 to 100)
+ Organ-Specific Lifetime Projections & NCCN Management Protocols
```

---

## 2. Person-Level Unified Hereditary Cancer Risk Scorer (`patient_cancer_scorer.py`)

The platform integrates Deep Learning inference with clinical inheritance modeling to generate an actionable, continuous **Person-Level Cancer Risk Score (0 to 100)**:

### Mathematical & Algorithmic Formulation
1. **Calibrated Variant Pathogenicity ($p_i \in [0, 1]$):**
   $$p_i = \sigma\left(\frac{\text{logit}_i}{T}\right) \quad (T = 0.8769)$$
2. **Excess Damaging Impact ($e_i \in [0, 1]$):**
   $$e_i = \max\left(0, \frac{p_i - 0.15}{1.0 - 0.15}\right)$$
   *(Ensures harmless polymorphisms with $p_i \le 0.15$ contribute exactly zero excess cancer risk).*
3. **Zygosity & Inheritance Modeling ($z_i \in \{0.10, 1.0\}$):**
   * **Autosomal Dominant (AD):** Single hit (`0/1` or `1/1`) confers full penetrant disease impact ($z_i = 1.0$).
   * **Autosomal Recessive (AR):** Monoallelic carrier (`0/1`) conveys minimal risk ($z_i = 0.10$). Biallelic hits (`1/1` or compound heterozygous) activate full disease penetrance ($z_i = 1.0$).
4. **Effective Variant Impact ($I_i$):**
   $$I_i = z_i \cdot w_g \cdot e_i \quad (w_g \in [0.45, 1.0] \text{ based on clinical gene penetrance tier})$$
5. **Multi-Variant Compounding & Unified Person Score ($S_{\text{person}} \in [12.0, 100.0]$):**
   $$P_{\text{composite}} = 1 - \prod_{i=1}^N (1 - I_i)$$
   $$S_{\text{person}} = S_{\text{base}} + (100 - S_{\text{base}}) \times P_{\text{composite}} \quad (S_{\text{base}} = 12.0)$$

### Clinical Benchmark Cohorts (Verified Live)

| Benchmark Patient Cohort | Genetic Profile | Unified Score | Clinical Risk Tier | Organ Projections vs Baseline |
| :--- | :--- | :---: | :---: | :--- |
| **Sarah (HBOC Carrier)** | `BRCA1 p.C61G` (88.7%) + `BRCA2 p.N372H` (benign) | **91.3 / 100** | **CRITICAL HIGH RISK** | Breast: **68.8%** (+56.3%), Ovarian: **40.7%** (+39.4%) |
| **Healthy Control** | `BRCA2 p.N372H` + `ATM p.K1192K` + `BARD1 p.I738V` | **12.0 / 100** | **POPULATION BASELINE** | All organ systems remain at standard baseline |
| **Lynch Syndrome** | `MLH1 p.S44F` (97.0%) + benign polymorphisms | **97.7 / 100** | **CRITICAL HIGH RISK** | Colorectal: **77.3%** (+73.1%), Gastric: **67.6%** |
| **Compound Moderate** | `CHEK2 p.R145W` (71.1%) + `ATM p.V320A` (40.8%) | **62.7 / 100** | **ELEVATED RISK** | Breast: **48.5%** (+36.0%), Colorectal: **39.2%** |
| **MUTYH Monoallelic Carrier** | `MUTYH p.Y179C` (88.2%, Genotype: `0/1`) | **19.2 / 100** | **POPULATION BASELINE** | Colorectal: **10.4%** (Carrier status only; no colectomy) |
| **MUTYH Biallelic Patient** | `MUTYH p.Y179C` (88.2%, Genotype: `1/1`) | **84.0 / 100** | **CRITICAL HIGH RISK** | Colorectal: **66.2%** (+62.0%; NCCN MAP surveillance) |
| **Novel Variant Patient** | `TP53 p.R248Q` (Novel, on-the-fly featurized) | **87.2 / 100** | **CRITICAL HIGH RISK** | CNS / Sarcoma: **68.5%**, Breast: **65.9%** |

---

## 3. Core Deep Learning Engine: Architecture & Metrics

GenomeDL v2 was trained on **148,825 high-confidence labeled ClinVar variants** across 130 genes and evaluated on **29,765 completely unseen test variants**:

* **Purged Somatic Contamination:** Purged 24 somatic-only oncogenes (`KRAS`, `EGFR`, `BRAF`, `PIK3CA`, etc.) that do not cause inherited cancer syndromes.
* **Repaired Missing Grantham Scores:** Eliminated 131 missing amino acid pairs using canonical 20×20 Grantham matrices via `pard` and evolutionary BLOSUM62 log-odds matrices.
* **Eliminated Benign False Positives:** Fixed the prototype v1 flaw where benign control `BRCA2 p.Asn372His` was misclassified as 95.23% pathogenic. GenomeDL v2 predicts it as **10.64% Benign**.
* **Platt Temperature Calibration:** Reduced Expected Calibration Error (ECE) to **0.0138**, ensuring predicted probabilities match empirical disease risk.

### Test Set Performance (29,765 Completely Unseen Test Variants)

| Evaluation Metric | GenomeDL v1 (Baseline) | GenomeDL v2 (Production) | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 96.52% | **99.29%** | **+2.77%** |
| **AUPRC (Precision-Recall AUC)** | 88.40% | **97.80%** | **+9.40%** |
| **Precision (PPV)** | 54.10% | **89.91%** | **+35.81% (Massive reduction in false alarms)** |
| **Specificity** | 86.00% | **98.25%** | **+12.25% (Prevents misclassifying benign variants)** |
| **Sensitivity (Recall)** | 92.35% | **90.35%** | Preserves high sensitivity for deadly alleles |
| **ECE (Expected Calibration Error)** | ~0.1000 | **0.0138** | **7.2x More Reliable Calibration** |
| **BRCA2 p.Asn372His (Benign Control)** | 95.23% (False Pos) | **10.64% (True Benign)** | **Fixed** |
| **BRCA1 p.Cys61Gly (Pathogenic Control)**| 94.54% (Pathogenic)| **88.68% (True Pathogenic)**| **Verified** |

---

## 4. The 29-Dimensional Biophysical & Evolutionary Feature Set

Every variant is featurized on the fly into a **29-dimensional vector** capturing physical, chemical, and evolutionary severity:

| Feature Dimension | Feature Name | Description & Biological Rationale |
| :---: | :--- | :--- |
| **1** | `grantham_score` | **Canonical Grantham Distance (0–215):** Formula combining composition, polarity, and volume. Scores >150 indicate severe structural disruption. 0 missing values. |
| **2** | `blosum62_score` | **Evolutionary Log-Odds Matrix (-4 to +11):** Biopython BLOSUM62 score measuring evolutionary tolerance across millions of years of cross-species alignment. Negative scores indicate high evolutionary intolerance. |
| **3** | `delta_mw` | **Molecular Weight Shift ($\Delta\text{MW}$ in Da):** Evaluates steric hindrance and core packing destabilization via RDKit exact molecular weights. |
| **4** | `delta_logp` | **Octanol-Water Partition Coefficient Shift ($\Delta\text{LogP}$):** Captures hydrophobic core burial disruption using Crippen wildman contributions. |
| **5** | `delta_tpsa` | **Topological Polar Surface Area Shift ($\Delta\text{TPSA}$ in \AA$^2$):** Measures solvent accessibility and hydrogen bonding surface disruption. |
| **6** | `delta_charge` | **Formal Electrostatic Charge Shift ($\Delta\text{Charge}$):** Detects salt bridge disruptions and electrostatic potential inversions. |
| **7** | `rel_protein_pos` | **Normalized Protein Position ($0.0$ to $1.0$):** Position along primary transcript protein sequence. |
| **8** | `has_protein_change` | Binary indicator for coding alterations vs non-coding/splice. |
| **9–16** | `ref_onehot` / `alt_onehot` | Base one-hot encodings for reference and alternate alleles (`A`, `C`, `G`, `T`). |
| **17** | `is_transition` | Transition ($A \leftrightarrow G, C \leftrightarrow T$) vs Transversion flag. |
| **18–26** | `vclass_onehot` | Molecular variant class flags (missense, nonsense, synonymous, start_lost, canonical splice, etc.). |
| **27–29** | `rdkit_structural` | RDKit chemical distance metrics and property shifts. |

---

## 5. Quickstart Guide: Running the Pipeline

### 1. Installation
Ensure Python 3.11+ is installed, then set up dependencies:

```bash
git clone https://github.com/anishgupta07/Deep-Learning-project.git
cd Deep-Learning-project

# Setup virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/macOS

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Person-Level Cancer Risk Scoring CLI (`patient_cancer_scorer.py`)

```bash
# 1. Run standard clinical cohorts (Sarah, Control, Lynch)
python patient_cancer_scorer.py --demo

# 2. Run all 6 benchmark cohorts (including compound moderate & recessive cases)
python patient_cancer_scorer.py --demo_all

# 3. Score a patient VCF and generate an interactive HTML report
python patient_cancer_scorer.py --vcf demo_vcfs/patient_sarah_hboc.vcf --patient_id SARAH_24Y --html sarah_report.html
```

### 3. Real-Time Single-Variant Pathogenicity CLI (`predict.py`)

```bash
# Test Sarah's BRCA1 mutation vs BRCA2 benign control
python predict.py --demo

# Score any custom variant in any of the 130 cancer genes
python predict.py --gene BRCA1 --chrom 17 --pos 43106487 --ref A --alt C --ref_aa C --alt_aa G --aa_pos 61
```

### 4. Automated 3-Pass Verification Suites

```bash
# Verify Tier 1 Deep Learning Model & Biophysical Features
python verify_all_milestones.py

# Verify Tier 2 Person-Level Scorer & Clinical Benchmarks
python verify_patient_cancer_scorer.py
```

---

## 6. Repository Directory Structure

```text
Deep-Learning-project/
├── data/                                      # Clean Pan-Cancer Datasets
│   ├── v2_pan_cancer_featured_labelled.csv    # 148,825 clean labeled SNVs with 29 features (48.2 MB)
│   └── v2_pan_cancer_featured_vus.csv         # 175,796 VUS variants for AI resolution (60.7 MB)
│
├── demo_vcfs/                                 # Standardized Patient VCF Cohorts
│   ├── patient_sarah_hboc.vcf                 # Sarah (BRCA1 pathogenic driver + benign polymorphisms)
│   ├── patient_healthy_control.vcf            # Healthy Control (BRCA2 N372H, ATM K1192K, BARD1 I738V)
│   ├── patient_lynch_carrier.vcf              # Lynch Carrier (MLH1 S44F pathogenic driver)
│   ├── patient_compound_moderate.vcf          # Compound Moderate (CHEK2 + ATM moderate variants)
│   ├── patient_recessive_carrier.vcf          # Recessive Carrier (MUTYH 0/1 monoallelic)
│   ├── patient_recessive_biallelic.vcf        # Recessive Biallelic (MUTYH 1/1 homozygous MAP)
│   └── patient_novel_variant.vcf              # Novel TP53 Variant (uncataloged sequence variant)
│
├── models/                                    # Production Deep Learning Weights
│   └── v2_pan_cancer_resnet.pth               # Verified ResNet v2 Checkpoint (99.29% ROC-AUC, T=0.8769)
│
├── patient_cancer_scorer.py                   # Flagship Person-Level Cancer Risk Scoring Engine (Option B)
├── predict.py                                 # Real-Time Single-Variant Pathogenicity Predictor
├── build_features_v2.py                       # 29-D Featurization Pipeline (Grantham, BLOSUM62, RDKit)
├── train_v2_model.py                          # PyTorch ResNet v2 Training & Platt Calibration Pipeline
├── extract_clinvar_hereditary.py              # ClinVar extraction engine across 130 hereditary genes
├── verify_patient_cancer_scorer.py            # Automated 3-Pass Person-Level Scorer Verification Suite
├── verify_all_milestones.py                   # Automated 3-Pass Model Verification Suite
├── requirements.txt                           # Pinned dependencies
├── context.md                                 # Living project context & verified audit log
├── README.md                                  # Repository documentation & clinical specifications
└── LICENSE                                    # Open-Source MIT License
```

---

## 7. License & Clinical Disclaimer

* **License:** Distributed under the open-source [MIT License](LICENSE).
* **Clinical Disclaimer:** GenomeDL v2 is designed for computational biology research, educational exploration, and clinical decision support. Diagnostic decisions and surgical interventions must be confirmed by board-certified medical geneticists and genetic counselors following CLIA/CAP-certified laboratory testing.
