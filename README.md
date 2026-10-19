# GenomeDL v2: Deep Learning Pathogenicity Engine for 130 Hereditary Cancer Genes

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B%20%7C%20CUDA%20%7C%20CPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-2D9CDB)](https://www.rdkit.org/)
[![Biopython](https://img.shields.io/badge/Biopython-BLOSUM62-00599C)](https://biopython.org/)
[![ROC-AUC](https://img.shields.io/badge/Test%20ROC--AUC-99.29%25-2ECC71)](train_v2_model.py)
[![Precision](https://img.shields.io/badge/Precision-89.91%25-brightgreen)](train_v2_model.py)
[![Specificity](https://img.shields.io/badge/Specificity-98.25%25-brightgreen)](train_v2_model.py)
[![ECE Calibration](https://img.shields.io/badge/ECE%20Calibration-0.0138-blue)](train_v2_model.py)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end computational genomics and deep learning framework designed to evaluate constitutional genetic mutations across **130 validated hereditary cancer predisposition genes**, resolve Variants of Uncertain Significance (VUS), and predict standardized **ACMG Pathogenicity Classes** using biophysical chemistry, evolutionary conservation, and a calibrated Deep Tabular Residual Network.

---

## 1. Problem Statement: Resolving the "VUS Crisis" in Hereditary Cancer

### The Genomic Challenge
The human genome contains **3.2 billion base pairs**. When constitutional DNA is sequenced from blood or saliva, an individual exhibits **3.5 to 4.5 million natural genetic variants**. 

In hereditary cancer genetics, clinical gene panels target high- and moderate-penetrance tumor suppressors (*BRCA1, BRCA2, TP53, MLH1, MSH2, APC, PTEN, CDH1, PALB2, ATM*, etc.). However, in **30% to 40% of patient genetic tests**, sequencers identify a **Variant of Uncertain Significance (VUS)**—a rare missense mutation where doctors lack clinical case reports to know whether the mutation destroys protein function or is a harmless natural polymorphism.

```
3.2 Billion Base Pairs (Human DNA)
      │
      ▼
4 Million Natural Variants per Person (99.9% harmless human diversity)
      │
      ▼  [The 130-Gene Coordinate Filter]
15 to 35 Constitutional Protein-Altering Mutations
      │
      ▼  [GenomeDL ResNet v2 Engine]
Standardized ACMG 5-Class Pathogenicity Verdict (0: Benign ──▶ 4: Pathogenic)
```

### The Architectural Focus: Dedicated Molecular Variant Pathogenicity
To maintain absolute scientific rigor, GenomeDL follows the architectural paradigm established by premier research labs:
* **Google DeepMind's AlphaMissense** (*Science*, 2023)
* **Illumina's PrimateAI-3D** (*Science*, 2023)
* **Harvard's EVE** (*Nature*, 2021)

ClinVar is the global gold standard for **variant pathogenicity**, but it does not contain longitudinal patient follow-up cohorts to directly train individual cancer incidence models. Rather than relying on uncalibrated heuristic tables, GenomeDL focuses 100% on the core biophysical problem: **learning the universal physical laws of protein folding damage and evolutionary intolerance to classify variant pathogenicity with clinical precision.**

*(Note: An exploratory Tier-2 patient risk prototype is preserved in `docs/archive/tier2_patient_engine_prototype/` for methodological reference.)*

---

## 2. Verified Benchmark Results & Audit Corrections

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

## 3. The 29-Dimensional Biophysical & Evolutionary Feature Set

Every variant is featurized on the fly into a **29-dimensional vector** capturing physical, chemical, and evolutionary severity:

| Feature Dimension | Feature Name | Description & Biological Rationale |
| :---: | :--- | :--- |
| **1** | `grantham_score` | **Canonical Grantham Distance (0–215):** Formula combining composition, polarity, and volume. Scores >150 indicate severe structural disruption. 0 missing values. |
| **2** | `blosum62_score` | **Evolutionary Log-Odds Matrix (-4 to +11):** Biopython BLOSUM62 score measuring evolutionary tolerance across millions of years of cross-species alignment. Negative scores indicate high evolutionary intolerance. |
| **3** | `delta_mw` | **Steric Clash ($\Delta\text{MW}$ in Daltons):** Molecular weight difference between original and mutant residue. Bulky substitutions in tight pockets disrupt tertiary packing. |
| **4** | `delta_logp` | **Hydrophobic Core Disruption ($\Delta\text{LogP}$):** Shift in octanol-water partition coefficient. Swapping hydrophobic residues in protein cores causes unfolding. |
| **5** | `delta_tpsa` | **Topological Polar Surface Area ($\Delta\text{TPSA}$):** Change in surface area capable of forming hydrogen bonds. Disrupts catalytic active sites. |
| **6** | `delta_charge` | **Ionic Charge Shift ($-2.0$ to $+2.0$):** Measures gain or loss of electrical charge. Disrupts ionic salt bridges essential for tertiary fold stability. |
| **7** | `tanimoto_chem_dist` | **Chemical Topology Distance ($0.0$ to $1.0$):** RDKit Morgan circular fingerprint distance measuring chemical sidechain dissimilarities. |
| **8** | `rel_protein_pos` | **Normalized Protein Position ($0.0$ to $1.0$):** Codon position divided by total protein length, bounded strictly to $[0, 1]$. |
| **9** | `is_transition` | **Nucleotide Transition Indicator:** Purine-to-purine ($A \leftrightarrow G$) or pyrimidine-to-pyrimidine ($C \leftrightarrow T$) substitution. |
| **10–13** | `ref_A, ref_C, ref_G, ref_T` | **Wild-Type Reference Nucleotide:** One-hot indicator of starting genomic base. |
| **14–17** | `alt_A, alt_C, alt_G, alt_T` | **Mutant Alternate Nucleotide:** One-hot indicator of replacement base. |
| **18–26** | `mol_consequence_*` | **Molecular Consequence Classes:** One-hot encoding of functional consequence (missense, nonsense, synonymous, start lost, splice canonical, splice region, intronic, UTR). |
| **27** | `has_protein_change` | **Coding Alteration Flag:** Binary indicator of whether an amino acid substitution occurs. |
| **28** | `gene_id` | **Entity Embedding (16-D):** Learnable embedding capturing gene-specific essentiality, haploinsufficiency, and tolerance across all 130 genes. |
| **29** | `cancer_syndrome_id` | **Syndromic Lineage Embedding (8-D):** Maps variants to clinical hereditary cancer syndrome lineages. |

---

## 4. Supported 130 Hereditary Cancer Predisposition Genes

The platform screens 130 high- and moderate-penetrance genes across 17 distinct syndromic categories:

| Clinical Syndrome Category | Key Predisposition Genes | Associated Primary Malignancies |
| :--- | :--- | :--- |
| **Hereditary Breast & Ovarian (HBOC)** | `BRCA1`, `BRCA2`, `PALB2`, `ATM`, `CHEK2`, `BARD1`, `BRIP1`, `RAD51C`, `RAD51D`, `RAD50`, `NBN`, `MRE11A` | Breast, Ovarian, Pancreatic, Prostate |
| **Lynch Syndrome (Mismatch Repair)** | `MLH1`, `MSH2`, `MSH6`, `PMS2`, `EPCAM`, `MLH3`, `MSH3`, `PMS1` | Colorectal, Endometrial (Uterine), Gastric, Ovarian |
| **Polyposis Syndromes (FAP / MAP)** | `APC`, `MUTYH`, `NTHL1`, `POLE`, `POLD1`, `AXIN2`, `BMPR1A`, `SMAD4`, `GREM1` | Colorectal Adenomas, Duodenal, Gastric |
| **Li-Fraumeni Syndrome** | `TP53`, `CHEK1` | Soft-Tissue Sarcomas, Osteosarcomas, Breast, Brain, Adrenocortical |
| **Cowden / PTEN Hamartoma** | `PTEN`, `ENG` | Breast, Follicular Thyroid, Endometrial, Renal |
| **Hereditary Diffuse Gastric Cancer** | `CDH1`, `CTNNA1` | Diffuse Gastric, Lobular Breast |
| **Multiple Endocrine Neoplasia (MEN)** | `MEN1`, `RET`, `MAX`, `TMEM127`, `SDHA`, `SDHB`, `SDHC`, `SDHD`, `SDHAF2` | Parathyroid, Pituitary, Medullary Thyroid, Pheochromocytoma |
| **Von Hippel-Lindau & Renal Syndromes** | `VHL`, `FH`, `FLCN`, `MET` | Clear Cell Renal, Pheochromocytoma, Hemangioblastoma |
| **BAP1 Tumor Predisposition** | `BAP1` | Uveal Melanoma, Malignant Mesothelioma, Cutaneous Melanoma, Renal |
| **Hereditary Retinoblastoma** | `RB1` | Retinoblastoma, Pinealoblastoma, Osteosarcoma |
| **Wilms Tumor Predisposition** | `WT1`, `DIS3L2`, `CTR9`, `REST` | Nephroblastoma (Wilms Tumor) |
| **Fanconi Anemia Pathway** | `FANCA`, `FANCB`, `FANCC`, `FANCD2`, `FANCE`, `FANCF`, `FANCG`, `FANCI`, `FANCL`, `FANCM` | Acute Myeloid Leukemia, Squamous Cell Carcinomas |
| **Neurofibromatosis & Schwannomatosis** | `NF1`, `NF2`, `SMARCB1`, `LZTR1` | Neurofibromas, MPNST, Vestibular Schwannomas |
| **Tuberous Sclerosis Complex** | `TSC1`, `TSC2` | Renal Angiomyolipomas, Subependymal Giant Cell Astrocytomas |
| **Melanoma & Pancreatic Predisposition** | `CDKN2A`, `CDK4`, `BAP1`, `MITF` | Cutaneous Malignant Melanoma, Pancreatic Adenocarcinoma |
| **Hereditary Prostate Cancer** | `HOXB13` | Early-Onset Aggressive Prostate Adenocarcinoma |
| **Ataxia-Telangiectasia & Bloom / Werner**| `ATM`, `BLM`, `WRN`, `RECQL4` | Lymphomas, Leukemias, Sarcomas |

---

## 5. Quickstart Guide: Running the Pipeline

### 1. Environment Installation
Ensure Python 3.11+ is installed, then set up dependencies:

```bash
# Clone the repository
git clone https://github.com/anishgupta07/Deep-Learning-project.git
cd Deep-Learning-project

# Setup virtual environment
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate # Linux/macOS

# Install pinned dependencies
pip install -r requirements.txt
```

### 2. Fast Real-Time Single-Variant Pathogenicity CLI
Evaluate any mutation on the fly in $<15\text{ ms}$:

```bash
# Run automated validation test on Sarah's BRCA1 mutation vs BRCA2 benign control
python predict.py --demo

# Score a custom variant in any of the 130 cancer genes
python predict.py --gene BRCA1 --chrom 17 --pos 43106487 --ref A --alt C --ref_aa C --alt_aa G --aa_pos 61
```

### 3. Run the Automated 3-Pass External Verification Suite
```bash
python verify_all_milestones.py
```
* **Pass 1:** Verifies 148,825 rows, 48 columns, 29 features, 0 nulls, and bounded metrics.
* **Pass 2:** Verifies checkpoint architecture, 130 genes, Platt calibration ($T = 0.8769$), and test metrics.
* **Pass 3:** Executes real-time on-the-fly inference on biological controls (`BRCA1 p.C61G`, `BRCA2 p.N372H`, `ATM p.K1192K`, `MLH1 p.S44F`).

---

## 6. Repository Directory Structure

```text
Deep-Learning-project/
├── data/                                      # Clean Pan-Cancer Datasets
│   ├── v2_pan_cancer_featured_labelled.csv    # 148,825 clean labeled SNVs with 29 features (48.2 MB)
│   └── v2_pan_cancer_featured_vus.csv         # 175,796 VUS variants for AI resolution (60.7 MB)
│
├── models/                                    # Production Deep Learning Weights
│   └── v2_pan_cancer_resnet.pth               # Verified ResNet v2 Checkpoint (99.29% ROC-AUC, T=0.8769)
│
├── predict.py                                 # Flagship Real-Time Variant Pathogenicity Scorer
├── build_features_v2.py                       # 29-D Featurization Pipeline (Grantham, BLOSUM62, RDKit)
├── train_v2_model.py                          # PyTorch ResNet v2 Training & Platt Calibration Pipeline
├── extract_clinvar_hereditary.py              # ClinVar extraction engine across 130 hereditary genes
├── verify_all_milestones.py                   # Automated 3-Pass Independent Verification Suite
├── requirements.txt                           # Pinned dependencies (torch, rdkit, biopython, pard, sklearn)
├── context.md                                 # Living project context & verified audit log
├── README.md                                  # Repository documentation & clinical specifications
└── LICENSE                                    # Open-Source MIT License
```

---

## 7. License & Clinical Disclaimer

* **License:** Distributed under the open-source [MIT License](LICENSE).
* **Clinical Disclaimer:** GenomeDL v2 is designed for computational biology research, educational exploration, and clinical decision support. Diagnostic decisions and surgical interventions must be confirmed by board-certified medical geneticists and genetic counselors following CLIA/CAP-certified laboratory testing.
