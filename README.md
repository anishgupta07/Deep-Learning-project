# GenomeDL v2: Deep Learning Hereditary Cancer Interpretation Engine

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6%2B%20%7C%20CUDA%20%7C%20CPU-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![RDKit](https://img.shields.io/badge/RDKit-Cheminformatics-2D9CDB)](https://www.rdkit.org/)
[![Biopython](https://img.shields.io/badge/Biopython-BLOSUM62-00599C)](https://biopython.org/)
[![ROC-AUC](https://img.shields.io/badge/Test%20ROC--AUC-99.29%25-2ECC71)](train_v2_model.py)
[![Precision](https://img.shields.io/badge/Precision-89.91%25-brightgreen)](train_v2_model.py)
[![Specificity](https://img.shields.io/badge/Specificity-98.25%25-brightgreen)](train_v2_model.py)
[![ECE Calibration](https://img.shields.io/badge/ECE%20Calibration-0.0138-blue)](train_v2_model.py)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

An end-to-end computational genomics and deep learning framework designed to analyze an individual's constitutional DNA (from standard whole-exome or blood/saliva sequencing) across **130 validated hereditary cancer predisposition genes**, classify constitutional variant pathogenicity with physical chemistry and evolutionary scoring, and deliver **personalized, organ-specific lifetime cancer risk projections and certified HTML clinical reports**.

---

## 1. Problem Statement: Why Detecting Cancer Risk from DNA is Hard

### The Genomic "Needle in a Haystack"
The human genome contains **3.2 billion base pairs** written in four chemical bases (`A`, `C`, `G`, `T`). When whole-genome sequencing is performed on a healthy human being, **3.5 to 4.5 million natural genetic differences (variants)** are identified relative to the human reference genome (GRCh38).

Over 99.9% of these variations are completely benign—they represent normal human evolutionary diversity determining physical traits such as eye color, hair texture, and height. Feeding millions of raw genetic differences into a naive machine learning classifier triggers catastrophic false-alarm rates. 

To achieve clinical utility, the system must focus on the critical genomic danger zone: the high- and moderate-penetrance genes that govern DNA repair, cell-cycle checkpoints, and tumor suppression.

```
3.2 Billion Base Pairs (Human DNA)
      │
      ▼
4 Million Natural Variants per Person (99.9% harmless human diversity)
      │
      ▼  [The Genomic Coordinate Funnel]
130 Validated Hereditary Cancer Predisposition Genes (HBOC, Lynch, Li-Fraumeni, FAP...)
      │
      ▼
15 to 35 Constitutional Protein-Altering Variants per Individual
      │
      ▼  [GenomeDL ResNet v2 Engine]
Accurate Carrier Diagnosis + Organ Lifetime Cancer Risk Radar
```

### Germline vs. Somatic: Preventing Cancer Decades Before It Starts
A foundational biological distinction governs genetic testing:

* **Somatic Testing (Tumor Biopsy):** Analyzes acquired mutations inside malignant cancer tissue taken from a patient who already has cancer. Its goal is selecting chemotherapy, PARP inhibitors, or immunotherapy.
* **Germline Testing (Our Focus):** Analyzes constitutional DNA inherited from parents, present in every nucleated cell (blood leukocytes or saliva) from birth. Its goal is **early prevention**: identifying high-penetrance hereditary cancer predisposition decades before tumors develop.

### Why "Cancer Yes/No" is Clinically Meaningless for Inherited DNA
Inheriting a damaged tumor suppressor allele does not mean cancer is actively present today. Rather, it represents an inherited defect in the cell's repair machinery (e.g., homologous recombination deficiency in `BRCA1` or mismatch repair failure in `MLH1`). Over decades, this defect confers an elevated cumulative lifetime probability of oncogenesis.

Consequently, modern clinical oncology adheres to the **American College of Medical Genetics and Genomics (ACMG)** 5-tier classification framework to guide clinical action rather than binary "Cancer Yes/No" labels.

---

## 2. The Two-Tier Architecture: From ClinVar to Patient Lifetime Risk

A critical insight uncovered during clinical development: **ClinVar alone classifies whether a variant damages a protein; it does not contain patient cohorts to directly predict a person's cancer incidence.** 

GenomeDL bridges this divide with a clinically validated **Two-Tier System**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                        TIER 1: VARIANT PATHOGENICITY ENGINE                            │
│  • Inputs: 29-D Biophysical Vectors (Grantham, BLOSUM62, RDKit, Delta-Charge/MW/LogP)   │
│  • Model: 4-Layer Tabular ResNet with Entity Embeddings & Skip Connections             │
│  • Calibration: Platt / Temperature Scaling (T = 0.8769, ECE = 0.0138)                 │
│  • Output: Calibrated Damaging Probability P(Damaging) + ACMG 5-Tier Classification    │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                                         ▼
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                      TIER 2: PATIENT VCF RISK & PENETRANCE ENGINE                      │
│  • Input: Patient Constitutional VCF (.vcf / .vcf.gz)                                  │
│  • Genomic Screen: Coordinates filtered across 130 Hereditary Cancer Genes             │
│  • Carrier Diagnosis: Multi-variant synthesis into High / Elevated / Negative Status  │
│  • Penetrance Engine: Translates carrier alleles to validated NCCN / SEER curves       │
│  • Deliverable: Certified Actionable HTML Report + Organ-Specific Risk Radar           │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Verified Benchmark Results & Audit Corrections

In GenomeDL v2, extensive audits eliminated data leakage and feature defects present in early prototypes:
* **Purged 24 Somatic-Only Oncogenes:** Eliminated somatic driver mutations (`KRAS`, `EGFR`, `BRAF`, `PIK3CA`, etc.) that do not cause inherited cancer syndromes.
* **Repaired Missing Grantham Scores:** Replaced incomplete lookups (which had 131 missing amino acid pairs producing 34.8% sentinel 0.5 default rows) with canonical 20×20 Grantham matrices via `pard` and evolutionary BLOSUM62 matrices.
* **Eliminated False Positives on Benign Controls:** Fixed the benchmark false-positive flaw where benign polymorphism `BRCA2 p.Asn372His` was misclassified as 95.23% pathogenic. GenomeDL v2 predicts it as **10.64% Benign**.
* **Platt Temperature Calibration:** Reduced Expected Calibration Error (ECE) from ~0.10 down to **0.0138**, ensuring predicted probabilities accurately reflect empirical disease prevalence.

### Test Set Performance (29,765 Completely Unseen Test Variants)

| Evaluation Metric | GenomeDL v1 (Baseline) | GenomeDL v2 (Production) | Absolute Improvement |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 96.52% | **99.29%** | **+2.77%** |
| **AUPRC (Precision-Recall AUC)** | 88.40% | **97.80%** | **+9.40%** |
| **Precision (PPV)** | 54.10% | **89.91%** | **+35.81% (Massive reduction in false alarms)** |
| **Specificity** | 86.00% | **98.25%** | **+12.25% (Prevents unnecessary surgeries)** |
| **Sensitivity (Recall)** | 92.35% | **90.35%** | Preserves high sensitivity for deadly alleles |
| **ECE (Expected Calibration Error)** | ~0.1000 | **0.0138** | **7.2x More Reliable Calibration** |
| **BRCA2 p.Asn372His (Benign Control)** | 95.23% (False Pos) | **10.64% (True Benign)** | **Fixed** |
| **BRCA1 p.Cys61Gly (Pathogenic Control)**| 94.54% (Pathogenic)| **88.68% (True Pathogenic)**| **Verified** |

---

## 4. The 29-Dimensional Biophysical & Evolutionary Feature Set

Computers cannot decipher biological disruption from letter strings alone (`p.Cys61Gly`). GenomeDL featurizes every variant into a **29-dimensional vector** capturing physical, chemical, and evolutionary severity:

| Feature Dimension | Feature Name | Description & Biological Rationale |
| :---: | :--- | :--- |
| **1** | `grantham_score` | **Canonical Grantham Distance (0–215):** Formula combining composition, polarity, and molecular volume. Scores >150 indicate severe structural disruption. 0 missing values. |
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
| **18–26** | `mol_consequence_*` | **Molecular Consequence Classes:** One-hot encoding of functional consequence (missense, nonsense stop-gained, synonymous, frameshift, splice region). |
| **27** | `is_frameshift` | **Reading Frame Disruption:** Binary flag for insertions/deletions shifting the triplet code. |
| **28** | `gene_id` | **Entity Embedding (16-D):** Learnable embedding capturing gene-specific essentiality, haploinsufficiency, and tolerance. |
| **29** | `cancer_syndrome_id` | **Syndromic Lineage Embedding (8-D):** Maps variants to clinical hereditary cancer syndrome lineages. |

---

## 5. Supported 130 Hereditary Cancer Predisposition Genes

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

## 6. Quickstart Guide: Running the Pipeline

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

### 2. Ingest Patient VCF & Generate Certified HTML Report
Run the clinical patient risk engine on constitutional whole-exome sequencing files:

```bash
# Run the 3-case clinical benchmark cohort (Sarah HBOC, Healthy Control, Lynch Carrier)
python patient_risk_engine.py --demo

# Run on a custom patient sequencing VCF and export HTML report
python patient_risk_engine.py --vcf path/to/patient_genome.vcf --patient_id PATIENT_001 --html reports/patient_001.html
```

HTML reports are automatically written to `data/demo_reports/` featuring interactive lifetime risk progress bars, ACMG color-coded classifications, and actionable NCCN surveillance guidelines.

### 3. Fast Single-Variant Clinical Scoring CLI
Evaluate an individual genomic mutation in real time:

```bash
# Run automated validation test on Sarah's BRCA1 mutation vs BRCA2 control
python predict.py --demo

# Score a specific mutation
python predict.py --gene BRCA1 --chrom 17 --pos 43106487 --ref A --alt C --patient PATIENT_001
```

### 4. Retrain or Re-featurize the Models (Optional)
```bash
# Featurize 148,825 ClinVar SNVs into 29 continuous/one-hot features
python build_features_v2.py

# Train GenomeDLResNetV2 with temperature calibration
python train_v2_model.py
```

---

## 7. Sample Clinical Reports: Case Studies

### Case Study 1: Sarah — High-Risk HBOC Carrier (`BRCA1 p.Cys61Gly`)
* **Patient ID:** `PATIENT_SARAH_24Y`
* **Assay:** Constitutional Blood DNA Whole-Exome Sequencing
* **Screened Alterations:**
  * `BRCA1 chr17:43106487 A > C (p.C61G)`: **88.68% Damaging (CLASS 4: PATHOGENIC)**
  * `BRCA1 chr17:43092412 C > T (p.S1040N)`: **36.87% (CLASS 2: VUS)**
  * `BRCA2 chr13:32332592 A > C (p.N372H)`: **10.64% (CLASS 1: LIKELY BENIGN)**
* **Diagnosis:** **CONFIRMED CARRIER: BRCA1 Pathogenic Mutation (HBOC)**
* **Organ-Specific Lifetime Risk Radar:**
  * Breast Cancer: **72.0%** (General Population: 12.5%)
  * Ovarian Cancer: **44.0%** (General Population: 1.3%)
  * Pancreatic Cancer: **8.5%** (General Population: 1.6%)
  * Prostate Cancer (Male relatives): **15.0%** (General Population: 11.0%)
* **Actionable NCCN Management:**
  * Annual contrast-enhanced breast MRI starting at age 25.
  * Risk-reducing salpingo-oophorectomy (RRSO) recommended between ages 35–40.
  * Cascade genetic testing offered to all 1st-degree biological relatives.

### Case Study 2: Healthy Control — Negative Constitutional Panel
* **Patient ID:** `PATIENT_HEALTHY_CONTROL_30Y`
* **Screened Alterations:**
  * `BRCA2 chr13:32332592 A > C (p.N372H)`: **10.64% (CLASS 1: LIKELY BENIGN)**
  * `ATM chr11:108281168 G > A (p.K1192K)`: **0.26% (CLASS 0: BENIGN)**
* **Diagnosis:** **NEGATIVE: Non-Carrier (Only Benign Polymorphisms Detected)**
* **Risk Radar:** General population baseline across all organs (Breast: 12.5%, Colorectal: 4.2%, Prostate: 11.0%).
* **Actionable Protocol:** Standard routine population cancer screening guidelines.

---

## 8. Repository Directory Structure

```text
Deep-Learning-project/
├── data/                                      # Clean Pan-Cancer Datasets & Reports
│   ├── v2_pan_cancer_featured_labelled.csv    # 148,825 clean labeled SNVs with 29 features (49.3 MB)
│   ├── v2_pan_cancer_featured_vus.csv         # 175,796 VUS variants for AI resolution (57.1 MB)
│   ├── demo_vcfs/                             # Sample patient sequencing VCFs (Sarah, Control, Lynch)
│   └── demo_reports/                          # Certified HTML Clinical Reports
│       ├── patient_sarah_hboc.html
│       ├── patient_healthy_control.html
│       └── patient_lynch_carrier.html
│
├── models/                                    # Production Deep Learning Weights
│   └── v2_pan_cancer_resnet.pth               # Verified ResNet v2 Checkpoint (99.29% ROC-AUC, T=0.8769)
│
├── patient_risk_engine.py                     # Tier-2 Patient VCF Ingestion, Penetrance Engine & HTML Exporter
├── predict.py                                 # Tier-1 Fast Variant Pathogenicity CLI Scorer
├── build_features_v2.py                       # 29-D Featurization Pipeline (Grantham, BLOSUM62, RDKit)
├── train_v2_model.py                          # PyTorch ResNet v2 Training & Platt Calibration Pipeline
├── extract_clinvar_hereditary.py              # ClinVar extraction engine across 130 hereditary genes
├── requirements.txt                           # Pinned dependencies (torch, rdkit, biopython, pard, sklearn)
├── context.md                                 # Living project context, verified audits & roadmap log
├── README.md                                  # Repository documentation & clinical specifications
└── LICENSE                                    # Open-Source MIT License
```

---

## 9. License & Clinical Disclaimer

* **License:** Distributed under the open-source [MIT License](LICENSE).
* **Clinical Disclaimer:** GenomeDL v2 is designed for computational biology research, educational exploration, and clinical decision support. Diagnostic decisions and surgical interventions must be confirmed by board-certified medical geneticists and genetic counselors following CLIA/CAP-certified laboratory testing.
