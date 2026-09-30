# 11-Class Multi-Cancer Genomic Risk Prediction Pipeline
==========================================================

This module implements the end-to-end multi-class data harvesting, variant calling, and Deep Learning dataset pipeline across **11 major human cancer types**.

---

## 1. The 11 Cancer Classes & Pathological Subtypes

| Class ID | Cancer Category | Target Pathological Subtype |
| :---: | :--- | :--- |
| **0** | **Oral Cavity Carcinoma** | Oral Squamous Cell Carcinoma (OSCC) |
| **1** | **Lung Cancer** | Bronchogenic Carcinoma / Lung Adenocarcinoma |
| **2** | **Female Breast Cancer** | Breast Carcinoma (Invasive Ductal / Lobular) |
| **3** | **Colorectal Cancer** | Colorectal Adenocarcinoma |
| **4** | **Prostate Cancer** | Prostate Adenocarcinoma |
| **5** | **Stomach Cancer** | Gastric Adenocarcinoma |
| **6** | **Thyroid Cancer** | Thyroid Carcinoma |
| **7** | **Liver Cancer** | Hepatocellular Carcinoma (HCC) |
| **8** | **Bladder Cancer** | Urothelial Carcinoma |
| **9** | **Cervical Cancer** | Cervix Uteri Carcinoma |
| **10** | **Non-Hodgkin Lymphoma** | Lymphoid Malignancy (DLBCL) |

---

## 2. End-to-End Pipeline Execution Guide

### Step 1: Download the Multi-Cancer Reference Gene Panel
Searches NCBI Gene for the top driver genes across all 11 cancer types and compiles them into a unified multi-gene FASTA reference:
```bash
python download_multiclass_genes.py 50
```
- **Output**: `multiclass_cancer_reference_panel.fna` (~500-700 unique human cancer driver genes, ~40-60 MB).

---

### Step 2: Harvest Patient SRA Runs & Generate Multi-Class VCFs
Discovers balanced patient cohorts from the European Nucleotide Archive (ENA Portal API) across all 11 cancer classes, streams real FASTQ sequence reads, and calls variants:
```bash
python process_multiclass_sra_pipeline.py 20 multiclass_cancer_reference_panel.fna
```
- **Parameters**: `20` = 20 patients per cancer class ($20 \times 11 = 220$ patients total).
- **Disk Protection**: Automatically deletes temporary raw reads after each VCF is called, keeping storage usage under **3.4 GB**.
- **Output**: `vcf_output_multiclass/class_00_Oral_Cavity_Carcinoma_sample_0001_SRR...vcf`, `vcf_output_multiclass/class_01_Lung_Cancer_sample_0001_SRR...vcf`, etc.

---

### Step 3: Build the Multi-Class Master Deep Learning Dataset CSV
Converts all generated VCFs into the 21-bp one-hot encoded dataset matrix ($168$ sequence features + sequencing quality metrics + 11-class target labels):
```bash
python build_multiclass_master_dataset.py vcf_output_multiclass multiclass_cancer_reference_panel.fna multiclass_cancer_master_dataset.csv
```
- **Outputs**:
  - `multiclass_cancer_master_dataset.csv` (Unified matrix with `cancer_class_id`: 0 to 10).
  - `multiclass_dataset_schema.json` (Dataset schema and class distribution).

---

### Step 4: Train Multi-Class Neural Network Model
Trains a Multi-Class Multi-Layer Perceptron (MLP) with Softmax probabilities across all 11 cancer classes:
```bash
python train_multiclass_model.py multiclass_cancer_master_dataset.csv
```
- **Metrics**: Top-1 Accuracy, Multi-Class Precision, Recall, Macro/Micro F1-Score, and Confusion Matrix across all 11 cancer types!
