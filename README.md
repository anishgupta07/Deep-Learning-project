# One-Sample Genomic Variant Calling & Deep Learning Dataset Pipeline
========================================================================

Welcome to the **`one_sample`** genomic pipeline module! This folder contains a self-contained, reproducible pipeline that takes raw gene FASTA files and patient sequencing SRA files, performs variant calling against a combined reference genome panel, and exports a 21-bp one-hot encoded dataset matrix ready for Deep Learning (MLP, 1D-CNN, Transformer) training.

---

## 1. Quick Start: How to Run

### Prerequisites
Make sure you have Python 3.9+ and the standard scientific libraries installed:
```bash
pip install numpy pandas biopython scikit-learn
```

### Running the Entire Pipeline (One-Click)
Run the automated runner script from inside the `one_sample/` directory:

```bash
cd one_sample
python run_one_sample_pipeline.py
```

### What Happens Automatically:
1. **Gene Consolidation**: Finds all gene reference files in `raw_data/` and merges them into `combined_reference_genes.fna`.
2. **Variant Calling**: Aligns every patient SRA file in `raw_data/` against `combined_reference_genes.fna` and outputs standard `.vcf` files in `vcf_output/`.
3. **Deep Learning Feature Encoding**: Extracts 21-bp sequence context windows, builds 168 one-hot tensor columns, extracts genomic/sequencing metrics, and generates `one_sample_dataset.csv`.

---

## 2. Folder & File Structure

```
one_sample/
│
├── raw_data/                                 # Raw input genomic files
│   ├── gene_01_EGFR.fna                      # Reference Gene 1 (EGFR Oncogene, 193.6 KB)
│   ├── gene_02_ERBB2.fna                     # Reference Gene 2 (ERBB2 / HER2 Oncogene, 40.9 KB)
│   ├── patient_01_SRR27944968.fasta          # Patient 1 Sequencing Reads (18.31 MB, 89,347 reads)
│   └── patient_02_SRR30142027.fasta          # Patient 2 Sequencing Reads (18.98 MB, 93,448 reads)
│
├── vcf_output/                               # Individual patient outputs
│   ├── person_01_patient_01_SRR27944968.vcf  # VCF for Patient 1 (326.4 KB, 3,311 Variants)
│   ├── person_02_patient_02_SRR30142027.vcf  # VCF for Patient 2 (285.7 KB, 2,903 Variants)
│   ├── person_01_patient_01_SRR27944968_encoded.csv # Individual CSV for Patient 1 (3,311 rows)
│   ├── person_02_patient_02_SRR30142027_encoded.csv # Individual CSV for Patient 2 (2,903 rows)
│   └── *_schema.json                         # Feature schema specifications for PyTorch DataLoader
│
├── combined_reference_genes.fna              # Consolidated 2-Gene Reference Panel (234.4 KB)
├── one_sample_dataset.csv                    # UNIFIED DEEP LEARNING MASTER DATASET (6,214 Rows x 188 Cols)
├── run_one_sample_pipeline.py                # Automated Pipeline Runner Script
└── README.md                                 # This Documentation File
```

---

## 3. Detailed Explanation of the CSV Dataset (`one_sample_dataset.csv`)

The output file [`one_sample_dataset.csv`](file:///m:/cancer-test/one_sample/one_sample_dataset.csv) contains **6,214 rows** (3,311 variants from Patient 1 + 2,903 variants from Patient 2) and **188 feature columns**.

### Why 21-bp Sequence Window & One-Hot Encoding?
1. **Biological Context**: A genetic mutation rarely acts in isolation. The immediate 10 base pairs upstream and 10 base pairs downstream (21 bp total) dictate whether the mutation alters a transcription factor binding site, promoter motif, or splice donor/acceptor site.
2. **Numerical Tensor Representation**: Deep learning models cannot process raw text strings (`"A"`, `"C"`, `"G"`, `"T"`). We encode each nucleotide into a 4-dimensional binary vector:
   $$A = [1, 0, 0, 0], \quad C = [0, 1, 0, 0], \quad G = [0, 0, 1, 0], \quad T = [0, 0, 0, 1]$$

---

### Comprehensive Column-by-Column Dictionary (188 Columns)

#### Group 1: Metadata & Variant Identifiers (Cols 1–6)
| Column Name | Data Type | Description & Example |
| :--- | :---: | :--- |
| `variant_id` | `string` | Unique genomic identifier in `<chrom>_<pos>_<ref>_<alt>` format (e.g., `EGFR_GeneID1956_NC_000007.14_55023457_G_A`). |
| `chrom` | `string` | Reference chromosome / gene accession ID (e.g., `EGFR_GeneID1956_NC_000007.14`). |
| `pos` | `integer` | 1-based genomic coordinate of the mutation on the reference sequence (e.g., `55023457`). |
| `ref` | `string` | Reference healthy allele base(s) (e.g., `G`). |
| `alt` | `string` | Mutated alternate allele base(s) found in the patient sample (e.g., `A`). |
| `mutation_type` | `string` | Classification of mutation: `SNV` (Single Nucleotide Variant), `INS` (Insertion), `DEL` (Deletion). |

#### Group 2: Biological & Mutational Spectrum Features (Cols 7–11)
| Column Name | Data Type | Description & Example |
| :--- | :---: | :--- |
| `is_transition` | `binary (0/1)` | `1` if mutation is a Transition ($C \leftrightarrow T$ or $A \leftrightarrow G$), `0` if Transversion ($C/T \leftrightarrow A/G$). Transitions are the most common somatic mutational signature in human cancers. |
| `trinucleotide_context` | `string` | 3-bp sequence context representation in `Upstream[REF>ALT]Downstream` format (e.g., `T[G>A]C`). Used to identify COSMIC cancer mutational signatures. |
| `ref_seq_21bp` | `string` | Raw 21-bp reference DNA sequence window (10 bp upstream + REF + 10 bp downstream). |
| `alt_seq_21bp` | `string` | Raw 21-bp mutated patient DNA sequence window (10 bp upstream + ALT + 10 bp downstream). |
| `gc_content_21bp` | `float` | Fraction of Guanine (G) and Cytosine (C) bases in the 21-bp window ($0.0$ to $1.0$). High GC content indicates CpG islands and methylation hotspots. |

#### Group 3: Sequencing Quality & Genotype Metrics (Cols 12–17)
| Column Name | Data Type | Description & Example |
| :--- | :---: | :--- |
| `dp` | `integer` | Total Read Depth (total number of sequenced reads covering this genomic position, e.g., `12`). |
| `af` | `float` | Variant Allele Frequency ($AO / DP$). Proportion of reads supporting the mutation (e.g., $1.0000$ for homozygous, $0.5000$ for heterozygous). |
| `ro` | `integer` | Reference Allele Observation count (number of reads matching the healthy reference base). |
| `ao` | `integer` | Alternate Allele Observation count (number of reads matching the mutated base). |
| `qual` | `float` | Phred-scaled confidence score (e.g., $26.8$, $99.0$). Higher values indicate near-zero sequencing machine error probability. |
| `gt_code` | `integer (0,1,2)` | Numerical Genotype representation: `0` = $0/0$ (Homozygous Reference), `1` = $0/1$ (Heterozygous Alternate), `2` = $1/1$ (Homozygous Alternate). |

#### Group 4: Classification Target Labels (Cols 18–19)
| Column Name | Data Type | Description & Example |
| :--- | :---: | :--- |
| `label_high_qual` | `binary (0/1)` | `1` if variant passes stringent quality criteria ($QUAL \ge 30, DP \ge 10, AF \ge 0.20$), `0` otherwise. |
| `label_somatic_candidate` | `binary (0/1)` | `1` if variant displays somatic hotspot characteristics ($0.20 \le AF \le 0.80, DP \ge 15, QUAL \ge 30$), `0` otherwise. |

#### Group 5: One-Hot Sequence Tensors (Cols 20–187, 168 Columns Total)
- **Reference Sequence One-Hot (`one_hot_ref_0` to `one_hot_ref_83`)**:
  - $21 \text{ positions} \times 4 \text{ channels } (A, C, G, T) = \mathbf{84 \text{ binary features}}$.
  - Position 0: `one_hot_ref_0` ($A$), `one_hot_ref_1` ($C$), `one_hot_ref_2` ($G$), `one_hot_ref_3` ($T$).
  - Position 20: `one_hot_ref_80` to `one_hot_ref_83`.
- **Alternate Sequence One-Hot (`one_hot_alt_0` to `one_hot_alt_83`)**:
  - $21 \text{ positions} \times 4 \text{ channels } (A, C, G, T) = \mathbf{84 \text{ binary features}}$.
- **Total Sequence Tensor Columns**: **168 numerical columns**.

#### Group 6: Patient Sample Identifier (Col 188)
| Column Name | Data Type | Description & Example |
| :--- | :---: | :--- |
| `patient_id` | `string` | Sample identifier for patient-level grouping and multi-sample cohort analysis (e.g., `person_01_patient_01_SRR27944968` or `person_02_patient_02_SRR30142027`). |

---

## 4. How Teammates Can Build Ahead (Deep Learning Code Examples)

Teammates can directly load `one_sample_dataset.csv` into PyTorch, TensorFlow, or Scikit-Learn:

### Example: Training an MLP in PyTorch / Scikit-Learn

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, roc_auc_score

# 1. Load the Dataset
df = pd.read_csv('one_sample_dataset.csv')
print(f"Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# 2. Separate Sequence Features (168 cols) and Tabular Metrics
one_hot_cols = [c for c in df.columns if c.startswith('one_hot_')]
metric_cols = ['dp', 'af', 'ro', 'ao', 'qual', 'gt_code', 'gc_content_21bp', 'is_transition']

feature_cols = one_hot_cols + metric_cols
X = df[feature_cols].values
y = df['label_somatic_candidate'].values

# 3. Train / Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Multi-Layer Perceptron (MLP)
mlp = MLPClassifier(hidden_layer_sizes=(128, 64, 32), activation='relu', max_iter=200, random_state=42)
mlp.fit(X_train, y_train)

# 5. Evaluate
y_pred = mlp.predict(X_test)
y_prob = mlp.predict_proba(X_test)[:, 1]

print("=== MODEL EVALUATION REPORT ===")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}")
```

### Example: Reshaping for 1D-CNN (DeepBind Architecture)

```python
import torch

# Extract the 168 one-hot columns and reshape into (Batch Size, 4 Channels, 42 Sequence Length)
X_seq = df[one_hot_cols].values # shape: (N, 168)
X_seq_tensor = torch.tensor(X_seq, dtype=torch.float32).view(-1, 42, 4).transpose(1, 2)
# X_seq_tensor shape is now (N, Channels=4, Length=42), ready for nn.Conv1d!
```

---

## 5. Summary & Support

- All core scripts (`generate_vcf.py`, `build_dl_dataset.py`, `train_all_industry_models.py`) are located in the root repository.
- To add more genes or patient samples, simply drop additional `.fna` or `.fasta` files into `raw_data/` and re-run `python run_one_sample_pipeline.py`.
