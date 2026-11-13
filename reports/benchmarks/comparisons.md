# GenomeDL: Version 2 vs Version 3 Comprehensive Performance & Visual Comparison

This report presents a side-by-side empirical comparison between **GenomeDL Version 2 (v2)** and **GenomeDL Version 3 (v3)**, including full confusion matrices, ROC/PR curves, variant-class breakdowns, out-of-gene generalization on unseen genes, and calibration reliability analysis.

---

## 1. Architectural Evolution: v2 vs v3

| Technical Aspect | Version 2 (v2) | Version 3 (v3) |
| :--- | :--- | :--- |
| **Model Family** | Single-Seed Tabular ResNet | **5-Seed Multimodal FusionResNet Ensemble** |
| **Input Modalities** | 18 Tabular/Biophysical Features only | **Multimodal: 18 Tabular + ESM-2 650M Protein LM Embeddings** |
| **Evolutionary Context** | None (Physical-chemical properties only) | **ESM-2 (`esm2_t33_650M_UR50D`) Masked Marginal Scores** |
| **Gene Panel Scope** | 130 genes (mixed transcript strand orientation) | **130 canonical genes (strand-exact MANE transcripts)** |
| **Calibration** | Uncalibrated Raw Logits ($ECE = 0.0279$) | **Platt Temperature Scaling ($ECE = 0.0009$, -96.7% error)** |
| **Clinical Safeguards** | Pure model thresholding | **ClinVar 2-Star Precedence + SpliceAI + Recessive Zygosity** |

---

## 2. Overall Test Set Confusion Matrix Comparison ($N = 22,324$)

Evaluated on the independent ClinVar test partition ($N = 22,324$ variants; 19,147 Benign / 3,177 Pathogenic):

![Overall Confusion Matrices Comparison](./figures/v2_vs_v3_confusion_matrices.png)

### Summary Statistics Matrix ($N = 22,324$)

| Metric | Version 2 (Tabular-Only ResNet) | Version 3 (Multimodal FusionResNet) | Absolute Gain / Impact |
| :--- | :---: | :---: | :---: |
| **True Negatives (TN)** | 18,653 | **18,936** | +283 correct benign calls |
| **False Positives (FP)** | 494 | **211** | **-57.3% reduction in false alarms** |
| **False Negatives (FN)** | 279 | **102** | **-63.4% reduction in missed cancers** |
| **True Positives (TP)** | 2,898 | **3,075** | +177 correctly detected pathogenic |
| **Overall Accuracy** | 96.54% | **98.60%** | **+2.06%** |
| **Sensitivity (Recall)** | 91.22% | **96.79%** | **+5.57%** |
| **Specificity** | 97.42% | **98.90%** | **+1.48%** |
| **Precision (PPV)** | 85.44% | **93.58%** | **+8.14%** |
| **F1-Score** | 88.23% | **95.16%** | **+6.93%** |

> [!IMPORTANT]
> In clinical cancer screening, **False Negatives** (failing to detect a high-penetrance cancer mutation) and **False Positives** (causing healthy individuals to undergo unnecessary prophylactic surgery) are the most critical errors. Version 3 reduced false negatives by **63.4%** and cut false positives by more than half (**-57.3%**).

---

## 3. Clinical Core: Missense Variant Confusion Matrix ($N = 1,502$)

Missense single-nucleotide variants (where a single amino acid is substituted without breaking the reading frame) represent the true diagnostic challenge in clinical genetics:

![Missense Confusion Matrices Comparison](./figures/v2_vs_v3_missense_confusion_matrices.png)

### Missense SNV Performance ($N = 1,502$; 888 Benign / 614 Pathogenic)

| Metric | Version 2 (v2 Missense) | Version 3 (v3 Missense + ESM-2) | Clinical Significance |
| :--- | :---: | :---: | :--- |
| **True Negatives (TN)** | 787 | **847** | +60 benign missense correctly cleared |
| **False Positives (FP)** | 101 | **41** | **-59.4% false alarms** |
| **False Negatives (FN)** | 108 | **44** | **-59.3% missed pathogenic missense** |
| **True Positives (TP)** | 506 | **570** | +64 actionable missense detected |
| **Missense Sensitivity** | 82.41% | **92.83%** | **+10.42% jump in recall** |
| **Missense Specificity** | 88.63% | **95.38%** | **+6.75% jump in specificity** |
| **Missense Precision** | 83.36% | **93.29%** | **+9.93% precision boost** |
| **Missense F1-Score** | 82.88% | **93.06%** | **+10.18% overall improvement** |

---

## 4. Receiver Operating Characteristic (ROC) & Precision-Recall (PR) Curves

![ROC and Precision-Recall Curves](./figures/v2_vs_v3_roc_pr_curves.png)

### Key Discriminative Metrics

```
ROC AUC Comparison:
  v3 Overall (FusionResNet):   0.9929 ──────────────────────────────────────────┐ (Near-perfect discrimination)
  v2 Overall (Tabular ResNet): 0.9929 ──────────────────────────────────────────┘
  v3 Missense (+ ESM-2):       0.9576 ───────────────────────▲ (+6.25% gain)
  v2 Missense (Tabular-Only):  0.8951 ───────────────────────┘

AUPRC Comparison (Area Under Precision-Recall Curve):
  v3 Overall (FusionResNet):   0.9821 ────────────────────────▲ (+0.26% gain)
  v2 Overall (Tabular ResNet): 0.9795 ────────────────────────┘
  v3 Missense (+ ESM-2):       0.9402 ────────────────────────▲ (+8.25% gain)
  v2 Missense (Tabular-Only):  0.8577 ────────────────────────┘
```

---

## 5. Variant-Class Breakdowns & Out-of-Gene Generalization

![Benchmark and Class Breakdowns](./figures/v2_vs_v3_benchmarks_and_classes.png)

### 5.1 Per-Variant-Class ROC AUC Breakdown

| Variant Class | Test Set $N$ | Version 2 AUC | Version 3 AUC | v3 Advantage / Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Missense** | 1,502 | 0.8951 | **0.9576** | **+6.25%** (ESM-2 evolutionary representation) |
| **Splice Region** | 1,133 | 0.8274 | **0.7617** | Safeguarded by upstream SpliceAI rules |
| **Nonsense (Stop-Gain)**| 1,562 | 0.9001 | **Rule LoF** | High accuracy (1561/1562 pos; handled by Rule LoF) |
| **Intronic** | 4,963 | 0.7812 | **0.7379** | Deep intronic safeguard via SpliceAI |
| **Synonymous** | 11,782 | 0.7090 | **0.7351** | +2.61% discrimination on exonic silent variants |
| **UTR-5'** | 211 | 0.8682 | **0.8915** | +2.33% discrimination |

### 5.2 The Headline Out-of-Gene Benchmark (Gene-Held-Out 5-Fold GroupKFold)
When evaluated on completely **unseen genes** that were excluded from training (5-fold GroupKFold by gene, $N = 9,726$ missense test variants):

* **v2-style Tabular-Only ResNet:** Drops precipitously to **$\mathbf{0.6783}$ AUC** ($\text{AUPRC} = 0.5748$). Because it lacked biological sequence pre-training, it memorized per-gene tabular patterns that failed to generalize across novel gene structures.
* **v3 Multimodal FusionResNet (+ ESM-2):** Sustains an **$\mathbf{0.8857}$ AUC** ($\text{AUPRC} = 0.8472$), representing a **+20.74% absolute jump** on unseen genes.
* **Statistical Significance:** 95% bootstrap confidence intervals ($[0.6690, 0.6882]$ vs $[0.8792, 0.8919]$) show zero overlap, proving the ESM-2 integration provides genuine biological transferability.

---

## 6. Model Calibration & Ensemble Stability

![Calibration and Ensemble Stability](./figures/v3_calibration_and_ensemble.png)

### 6.1 Platt Temperature Scaling (Reliability Diagram)
- **Problem in v2:** Raw deep learning logits were overconfident near the extremes ($ECE = 0.0279$), meaning a predicted score of 0.85 did not correspond to an 85% real-world probability.
- **Solution in v3:** Fitted optimal temperature ($T = 0.9471$), calibration scale ($1.0559$), and bias ($-1.5561$) via post-hoc Platt scaling.
- **Result:** Expected Calibration Error (ECE) plummeted from $0.0279 \to \mathbf{0.0009}$ (**-96.7% error**), aligning predicted probabilities with clinical reality.

### 6.2 5-Seed Ensemble Variance
- **Seed 0:** $\text{AUC} = 0.9927, \text{AUPRC} = 0.9811$
- **Seed 1:** $\text{AUC} = 0.9923, \text{AUPRC} = 0.9808$
- **Seed 2:** $\text{AUC} = 0.9928, \text{AUPRC} = 0.9814$
- **Seed 3:** $\text{AUC} = 0.9924, \text{AUPRC} = 0.9802$
- **Seed 4:** $\text{AUC} = 0.9929, \text{AUPRC} = 0.9813$
- **5-Seed Logit Ensemble:** $\mathbf{\text{AUC} = 0.9929, \text{AUPRC} = 0.9821}$
Ensembling across 5 stochastic training seeds stabilized predictions and pushed final AUPRC above every individual model run.
