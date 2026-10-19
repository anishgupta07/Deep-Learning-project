# Project Context: Germline Cancer Risk Deep Learning Engine (GenomeDL v2)

> **Last Updated:** 2026-10-19  
> **Repository:** `Deep-Learning-project`  
> **Status:** Focused 100% on Tier 1 (Deep Learning Variant Pathogenicity Engine). All Verifications Passing.

---

## 1. Project Goal & Strategic Architectural Focus
* **Primary Objective:** Build a state-of-the-art Deep Learning model for **constitutional variant pathogenicity prediction across 130 hereditary cancer predisposition genes**.
* **The Strategic Focus on Tier 1:**
  * ClinVar is a variant damage catalog, not a longitudinal patient cohort. 
  * Following the scientific paradigm of **Google DeepMind's AlphaMissense** (*Science*, 2023) and **Illumina's PrimateAI-3D** (*Science*, 2023), the repository focuses strictly on what machine learning can rigorously and reliably solve: **learning the physical, chemical, and evolutionary laws of protein disruption caused by inherited mutations**.
  * The exploratory Tier-2 patient penetrance rules engine has been archived in `docs/archive/tier2_patient_engine_prototype/` to preserve complete scientific integrity.

---

## 2. Core Deep Learning Engine (GenomeDL v2)

| Subsystem | Implementation Details | Verified Metric / Outcome |
| :--- | :--- | :---: |
| **Feature Space** | 29 continuous & categorical dimensions (`build_features_v2.py`): canonical Grantham, BLOSUM62, RDKit physicochemical shifts ($\Delta\text{MW}, \Delta\text{LogP}, \Delta\text{TPSA}, \Delta\text{Charge}$), Morgan fingerprints, bounded sequence position $[0, 1]$. | **0 null values across all 148,825 rows** |
| **Model Architecture** | `GenomeDLResNetV2` (`train_v2_model.py`): Entity embeddings for 130 genes and 15 syndromes, 4-layer residual blocks with skip connections ($\mathbf{x} + \mathcal{F}(\mathbf{x})$), LayerNorm, GELU, and Dropout. | **Checkpoint: `models/v2_pan_cancer_resnet.pth`** |
| **Calibration** | Platt Temperature Scaling layer ($T = 0.8769$) fitted on validation split. | **ECE: 0.0138 (Down from ~0.10)** |
| **Test Performance** | Evaluated on 29,765 completely unseen clinical test variants across all 130 genes. | **ROC-AUC: 99.29%**<br>**Precision: 89.91%**<br>**Specificity: 98.25%**<br>**Sensitivity: 90.35%** |
| **Control Fix** | Fixed prototype v1 false-alarm flaw on `BRCA2 p.Asn372His` (benign control). | **10.64% Benign (Fixed from 95.23%)** |

---

## 3. Flagship Artifacts & CLI Entrypoints

* **Production Checkpoint:** `models/v2_pan_cancer_resnet.pth` (1.2 MB, calibrated weights).
* **Master Datasets:**
  * `data/v2_pan_cancer_featured_labelled.csv` (148,825 labeled variants with 29 features).
  * `data/v2_pan_cancer_featured_vus.csv` (175,796 VUS variants for AI resolution).
* **Primary Real-Time CLI:**
  * `predict.py`: Evaluates any mutation in $<15\text{ ms}$ on the fly.
    ```bash
    python predict.py --demo
    python predict.py --gene BRCA1 --chrom 17 --pos 43106487 --ref A --alt C --ref_aa C --alt_aa G --aa_pos 61
    ```
* **Independent 3-Pass Audit Suite:**
  * `verify_all_milestones.py`: Complete automated test runner for external auditors.
    ```bash
    python verify_all_milestones.py
    ```

---

## 4. Verification Pass Results (100% Success)

```text
================================================================================
   GENOMEDL V2: DEEP LEARNING VARIANT PATHOGENICITY AUDIT SUITE
================================================================================
[*] Pass 1: Verified 148,825 rows, 48 cols, 29 features, 0 nulls, Grantham [0, 215], rel_protein_pos [0, 1].
[*] Pass 2: 130 Genes, 15 Syndromes, Platt Temperature = 0.8769, ROC-AUC 99.29%, Precision 89.91%, Specificity 98.25%, ECE 0.0138.
[*] Pass 3: Real-time on-the-fly inference:
    - Sarah BRCA1 p.C61G   : 88.68% (CLASS 4: PATHOGENIC)
    - Benign BRCA2 p.N372H : 10.64% (CLASS 1: LIKELY BENIGN)
    - Synonymous ATM p.K1192K: 0.26% (CLASS 0: BENIGN)
    - Lynch MLH1 p.S44F    : 97.03% (CLASS 4: PATHOGENIC)
>>> ALL 3 VERIFICATION PASSES COMPLETED WITH 100% SUCCESS <<<
```
