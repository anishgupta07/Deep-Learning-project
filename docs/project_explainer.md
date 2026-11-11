# GenomeDL — Project Explainer

*A plain-language guide to the whole project: the problem, the biology you need, how we solve it, the architecture, the train/test split and every feature. For exact verification numbers and known defects, see `README.md`.*

---

## 1. The problem in one minute

- Every person has about **4 million DNA differences** compared with the reference human genome. Almost all are harmless.
- A few are **dangerous** when they sit in genes that protect us from cancer (e.g. *BRCA1*, *TP53*, *MLH1*). People who inherit such a change have a much higher lifetime chance of cancer, so finding it early lets doctors screen early.
- Doctors need to know for each change: **harmful or harmless?** For many changes nobody knows yet. ClinVar calls these **VUS** (variant of uncertain significance).
- **Our goal:** build a machine-learning model that looks at one DNA change in one of 130 cancer-predisposition genes and predicts whether ClinVar would call it **pathogenic (1)** or **benign (0)**.

> **Honest framing.** The labels come from ClinVar curators, who themselves use rules ("a change that creates a stop codon is pathogenic", "a silent change is benign"). So the model learns *agreement with ClinVar*, not ground-truth biology. This is a **research prototype, not a clinical tool**.

---

## 2. Biology basics (everything you need)

| Term | Simple meaning |
|---|---|
| **DNA** | A long string of 4 letters: `A`, `C`, `G`, `T`. |
| **Gene** | A stretch of DNA that holds the recipe for one protein. |
| **Protein** | A chain of **amino acids**; its 3D shape does a job in the cell (e.g. repairing DNA). |
| **Codon** | 3 DNA letters = 1 amino acid (e.g. `TGT` → Cysteine). 64 codons, 20 amino acids + stop signals. |
| **Stop codon** | `TAA`, `TAG`, `TGA`: tells the cell to end the protein. |
| **Mutation / variant** | A change in the DNA letters. |
| **SNV** | Single-nucleotide variant: one letter swapped (e.g. `C → G`). This project only uses SNVs. |
| **Germline vs somatic** | Germline = inherited, present in every cell from birth (what we study). Somatic = acquired later inside a tumour. |
| **Hereditary cancer gene** | A gene where an inherited harmful variant raises cancer risk (the "broken brake pedal" idea). |
| **Strand (+ / −)** | DNA has two strands. Some genes are read on the plus strand, others on the minus strand, so the same change looks different (`A>C` on one strand = `T>G` on the other). Of our 130 genes, 65 are plus-strand and 65 minus-strand. |
| **Transcript** | One specific version of a gene's message (e.g. `NM_007294.4` for BRCA1). `c.181` means "base 181 of that transcript". |
| **GRCh38** | The reference human genome build used for all positions. |

### Types of SNV (the `variant_class` column)

| Class | What happens | Typical risk |
|---|---|---|
| **synonymous** | Letter changes, amino acid stays the same ("silent typo") | almost always benign |
| **missense** | One amino acid is replaced by another | **the hard, interesting case** (about 41 % pathogenic) |
| **nonsense** | A stop codon appears; the protein is cut short | ~100 % pathogenic |
| **start_lost** | The start codon (Met1) is destroyed | 100 % pathogenic in our data |
| **splice_canonical / splice_region** | Change at or near an exon–intron border; may break gene editing | canonical ≈ pathogenic; region is hard |
| **intronic, utr5, utr3, other_noncoding** | Change outside the coding part | mostly benign |

### How a variant is written (HGVS) — worked example
`NM_007294.4(BRCA1):c.181T>G (p.Cys61Gly)`
- `NM_007294.4` = transcript, `BRCA1` = gene.
- `c.181T>G` = at coding base 181, `T` became `G` (on the gene's own strand).
- `p.Cys61Gly` = amino acid 61 changed from Cysteine to Glycine. Because BRCA1 is a minus-strand gene, the plus-strand genome shows the same change as `A>C` at `chr17:43106487`.

### Labels and ClinVar
- **ClinVar** (NCBI) is a public database where labs and expert panels submit their classification of each variant.
- **Label 1** = pathogenic / likely pathogenic · **Label 0** = benign / likely benign · **VUS** = unknown (never used to train) · **conflicting** = labs disagree (excluded).
- **Review stars** show how well a call is supported: 1 = one lab with criteria, 2 = several labs agree, 3 = expert panel. We keep only ≥ 1 star.
- **ACMG/AMP** is the official 5-tier clinical system (benign → pathogenic). It is based on evidence rules, **not** on a probability cut-off.

---

## 3. The data we use

- **Source:** ClinVar `variant_summary.txt.gz` (GRCh38, germline only).
- **Panel:** 130 hereditary-predisposition genes in 15 groups (Breast/Ovarian repair, Lynch/mismatch repair, Li-Fraumeni, Polyposis, Renal, Fanconi, …).
- **Labelled SNV file (v2):** 148,825 rows = **127,657 benign + 21,168 pathogenic**.
- **Other files:** VUS (175,796, scoring only), conflicting (36,608, excluded), labelled indels (40,153, optional second task). *(these three: carried over from the earlier audit)*
- **Important property:** about 75 % of rows are synonymous or intronic (almost all benign) and nonsense/canonical-splice rows are almost all pathogenic. These are *easy*. The honest challenge is **missense (9,726 rows)** and **splice_region**.

---

## 4. How we solve it (the pipeline)

```
ClinVar raw file
   │  extract_clinvar_hereditary.py   (filter 130 genes, GRCh38, germline, 1-star+, label 0/1/VUS/conflict, parse HGVS → variant_class)
   ▼
clinvar_labelled_snv.csv  (+ manifest.json)
   │  build_v2_features.py            (protein position, Grantham, chemistry deltas, nucleotide one-hots, IDs)
   ▼
v2_clinvar_featured_snv.csv
   │  split 70 / 15 / 15  →  scale some columns  →  train ResNet (train_tabular_mlp.py)
   ▼
model → probability of "pathogenic"  →  evaluate on the untouched test set → (later) score VUS
```

**The idea:** a DNA letter change on its own tells a model little. What matters is the *damage to the protein*. So we turn each variant into numbers describing (a) what kind of change it is, (b) how different the new amino acid is physically and chemically, (c) where in the protein it sits, and (d) which gene it is in.

> **Status:** v2 data and features are built and verified. The training and prediction scripts are still **v1 code**; the neural network has **not yet been retrained on v2** (see `README.md` §6).

---

## 5. Features — what each one does (one line each)

### 5.1 The 19 continuous inputs of the model
| Feature | What it tells the model |
|---|---|
| `ref_A, ref_C, ref_G, ref_T` | Which DNA letter was originally there (one-hot, 4 columns). |
| `alt_A, alt_C, alt_G, alt_T` | Which DNA letter it became (one-hot, 4 columns). |
| `rel_protein_pos` | How far along the protein the change is (0 = start, 1 = end); ends are often more tolerant. |
| `is_transition` | 1 if the swap is purine↔purine or pyrimidine↔pyrimidine (`A↔G`, `C↔T`), the common, usually milder type. |
| `is_missense` | 1 if an amino acid is replaced by a different one. |
| `is_nonsense` | 1 if the change creates a stop codon (protein cut short). |
| `is_synonymous` | 1 if the amino acid does not change (silent). |
| `delta_mw` | Change in molecular weight, i.e. how much bigger or smaller the new amino acid is (steric clash). |
| `delta_logp` | Change in water-loving vs water-hating character; buried parts of proteins are water-hating. |
| `delta_tpsa` | Change in polar surface area, i.e. ability to form hydrogen bonds. |
| `delta_charge` | Change in electrical charge at body pH; charge swaps break salt bridges. |
| `tanimoto_chem_dist` | 0–1 chemical "unlikeness" of old and new amino acid from RDKit fingerprints. |
| `grantham_score` | Classic 1974 score (0–215) of how different two amino acids are in composition, polarity and volume. |

`delta_mw, delta_logp, delta_tpsa, delta_charge, grantham_score` are standardised (mean 0, spread 1) using **training data only**.

### 5.2 Categorical inputs (learned embeddings)
| Feature | What it does |
|---|---|
| `gene_id` | Gives each gene its own learnable 16-number "personality" (how fragile or tolerant that gene is). |
| `syndrome_id` | Same idea for the cancer-syndrome group, 8 numbers (it is fully determined by the gene, so partly redundant). |

### 5.3 Extra columns in the v2 featured CSV
| Column | What it does |
|---|---|
| `has_protein_change` | 1 if the variant changes a protein (missense, nonsense, synonymous, start-lost). |
| `variant_class_id` | Integer code for the 11 variant classes (synonymous, missense, nonsense, …); replaces the three `is_*` flags in v2. |

### 5.4 Descriptive columns (not fed to the model)
`variation_id` (ClinVar ID) · `gene`, `syndrome_group` · `chrom`, `pos`, `ref`, `alt` (GRCh38, plus strand) · `label` (the target) · `clinical_significance`, `review_status`, `stars` (evidence quality) · `ref_aa`, `alt_aa`, `aa_pos` (the protein change) · `transcript`, `on_primary_transcript` (which transcript the numbering belongs to) · `hgvs_name`, `last_evaluated` (raw ClinVar text).

---

## 6. Model architecture

A **tabular ResNet**: a small neural network with "skip connections" so the original signal is not washed out.

```
19 numeric features ─┐
gene_id  → Embedding → 16 numbers ─┤ glue together → 43 numbers
syndrome → Embedding →  8 numbers ─┘
        │
        ▼  Linear(43→256) → LayerNorm → GELU → Dropout
        ▼  Residual block 1   (output = GELU( x + F(x) ))
        ▼  Residual block 2
        ▼  Linear(256→64) → LayerNorm → GELU → Dropout → Linear(64→1)
        ▼  one number (logit) → sigmoid → probability of "pathogenic" (0 to 1)
```

- **Parameters:** 295,305 (for the v1 checkpoint with 93 genes; a v2 model needs 130 genes and 15 syndromes).
- **Loss:** weighted binary cross-entropy. Pathogenic rows are rarer (about 1 in 6), so each one counts about 6 times more.
- **Optimiser:** AdamW (learning rate 1e-3, weight decay 1e-4), cosine learning-rate schedule, 25 epochs, batch size 256, seed 42.
- **Output:** *one* probability. It is **not** a 5-class ACMG output; the "ACMG class" in `predict.py` is just arbitrary cut-offs on that probability.
- **Why a ResNet?** Skip connections keep strong raw signals (like a Grantham score of 200) intact. In practice, gradient boosting on the same features performs about as well (carried-over finding), so the architecture is a design choice, not a proven advantage.

---

## 7. How the train / validation / test split works

Each **row is one variant**, not one patient. The split is done on variants.

```
All labelled variants (~148,825 in v2)
   │
   ├── 70 %  TRAIN       the network learns from these
   ├── 15 %  VALIDATION  used during training to pick the best epoch (best ROC-AUC)
   └── 15 %  TEST        locked away; scored once at the end
```

**Exact recipe in the code (`train_tabular_mlp.py`)**
1. `train_test_split(test_size=0.30, stratify=label, seed=42)` → 70 % train and 30 % temporary.
2. Split the 30 % in half, again stratified → 15 % validation and 15 % test.
3. **Stratified** means every part keeps the same benign : pathogenic ratio (about 6 : 1), so no split is accidentally easy.
4. The `StandardScaler` is **fit on train only**, then applied to validation and test (no peeking).
5. After each epoch the model is scored on validation; the best-AUC checkpoint is saved and reloaded for the final test score.
6. For v1 (119,045 variants) this gave 83,331 / 17,857 / 17,857. For v2 expect roughly 104k / 22k / 22k.

VUS and conflicting variants are **never** used for training or testing.

### The weakness of this split (important)
- A random split puts **neighbouring variants in both train and test**: about 31 % of test missense variants share the exact same codon with a training variant, so the model can partly *memorise the location* instead of learning biology. *(measured in the earlier v1 audit)*
- Adding `gene_id` makes this stronger, because the model learns "variants in gene X are mostly pathogenic".
- Measured consequence (v1 audit, carried over): missense AUC ≈ **0.87–0.93** with random splits but only ≈ **0.71** when whole genes are held out.
- The overall AUC (0.9652 for the v1 model) is also inflated by easy rows: a lookup of just four numbers (the pathogenic rate per variant class) already reaches AUC 0.921.

### What to do instead (recommended for v2)
- **Split by gene** (e.g. `GroupKFold` on `gene`) so test genes are never seen in training. Report this as the headline.
- Report a protein-region split as a second, easier result.
- Report **AUC and AUPRC per variant class**, with **missense and splice_region** as the headline classes.
- Always compare against simple baselines (class prior, class + gene prior, gradient boosting with and without `gene_id`).
- Calibrate probabilities (isotonic or Platt) before reading them as risk.

---

## 8. How to read the results honestly

- **High overall AUC ≠ clinical usefulness.** Most rows are easy.
- **Precision matters:** at a 0.5 threshold the v1 model had about 54 % precision, so nearly half of its "pathogenic" calls were false alarms.
- **VUS scores** (about 91 % of VUS are missense, the model's weakest class) should be treated as hints for research, never as a diagnosis.
- **Not implemented:** reading patient VCF files, personalised lifetime risk, 3D protein structure, population frequency, conservation scores, calibrated ACMG classes.

---

## 9. Where to go next

1. Fix the two small data defects (79 rows with another gene's annotation, 20 stop-loss rows named missense) and rebuild.
2. Improve features: real protein lengths, coding-strand nucleotides, start-loss severity, gnomAD frequency, conservation (phyloP/GERP), AlphaMissense/REVEL/ESM scores, SpliceAI for splice variants.
3. Adapt `train_tabular_mlp.py` to the v2 columns and train with a gene-held-out split.
4. Only then score VUS and report results with confidence intervals.

*Maintainer: Anish Gupta · License: MIT*
