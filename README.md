# GenomeDL v3 — A DNA Spell-Checker for Cancer Genes

---

## What Does This Project Do? (The Big Picture)

Your DNA is like a **really long instruction manual** — about 3 billion letters long — that tells your body how to build and run itself. Sometimes, there's a **typo** in this manual. Most typos don't matter. But some typos in certain chapters (genes) can increase your chance of getting cancer.

**This project is a computer program that reads those typos and tells you: "This one is dangerous" or "This one is harmless."**

It focuses on **130 specific genes** that scientists already know are linked to hereditary cancer (like BRCA1, which you may have heard of in the context of breast cancer).

> ⚠️ **Important**: This is a research project, not a medical tool. If it finds something concerning, you'd still need a real doctor and a real lab to confirm it.

---

## How Does It Work? (The Simple Version)

Here's the entire project in 5 steps:

```
  YOUR DNA FILE           What typos          Are they              How bad         Here's your
  (a .vcf file)     →    are in cancer   →   dangerous or     →   is it really?  →   report
                          genes?              harmless?              (PDF)
```

Let's walk through each step:

### Step 1: Read the DNA file

Doctors sequence your DNA and save the results in a standard file format called a **VCF file**. Think of it like a spreadsheet where each row is one typo found in your DNA:

```
Chromosome 17, Position 43106487, the letter was A but yours says C
```

The program opens this file and pulls out only the typos in the 130 cancer genes it knows about. Everything else is ignored.

### Step 2: Quick checks first (no AI needed)

Before using any fancy AI, the program applies **common-sense rules**:

| Check | What it does | Example |
|---|---|---|
| **Is it super common?** | If more than 1% of all humans have this typo, it's almost certainly harmless. | "45% of people have this → Benign" |
| **Has a lab already reviewed it?** | Looks it up in ClinVar, a free database of previously classified variants. If expert labs agree, use their answer. | "ClinVar says Pathogenic (3 stars) → Pathogenic" |
| **Does it obviously break the gene?** | Some typos clearly destroy the protein (like deleting half of it). No model needed. | "Frameshift → Likely Pathogenic" |

**Only the leftover "unknown" typos** — the ones nobody has reviewed yet — go to the AI model.

### Step 3: The AI model scores each unknown typo

This is the deep learning part. The model looks at **three types of clues** about each typo:

#### Clue 1: Chemistry facts (numbers)
- How different are the original and new amino acids? (Size? Charge? Shape?)
- Where on the protein does the change happen?
- What kind of DNA letter change is it?

*Analogy: If you change a load-bearing wall in a house vs. painting a bedroom wall, the consequences are very different.*

#### Clue 2: Category labels
- What *type* of typo is it? (Does it swap one amino acid for another? Does it break the protein entirely? Is it in a non-coding region?)
- Which cancer syndrome group does this gene belong to?

#### Clue 3: What a protein language model thinks
There's a separate AI model called **ESM-2** (made by Meta) that has "read" millions of proteins. We ask it: *"At this position in this protein, is this new amino acid weird or normal?"*

If ESM-2 says "that amino acid would be very surprising here," that's a bad sign.

**The model combines all three clues and outputs a single number between 0 and 1:**
- Close to **1.0** = "I think this typo is dangerous"
- Close to **0.0** = "I think this typo is harmless"

### Step 4: Translate the score into a verdict

| Model score | Verdict |
|:---:|---|
| 0.90 or higher | **Pathogenic** (dangerous) |
| 0.70 – 0.89 | **Likely Pathogenic** (probably dangerous) |
| 0.30 – 0.69 | **Uncertain** (not enough evidence either way) |
| 0.10 – 0.29 | **Likely Benign** (probably harmless) |
| Below 0.10 | **Benign** (harmless) |

### Step 5: Generate a report

The program creates a clinical-style report that includes:
- **Which typos were found** and their verdicts
- **Lifetime cancer risk estimates** from published medical studies (e.g., "BRCA1 carriers have a 65–72% lifetime risk of breast cancer")
- A **PDF with charts** comparing your risk to the general population
- A **disclaimer** reminding you this needs real clinical confirmation

---

## The Training Data: Where Did the AI Learn?

The model learned from **ClinVar** — a public database where genetic testing labs around the world submit their findings. Each entry says something like:

> *"Gene BRCA1, position 61, Cysteine → Glycine: we reviewed this variant and it's **Pathogenic**."*

The project collected tens of thousands of these reviewed variants across the 130 cancer genes. Each one is labeled either:
- **Pathogenic** (label = 1) — known to cause disease
- **Benign** (label = 0) — known to be harmless

Variants where experts disagree or are uncertain were **excluded** from training.

### How the data was split

The data is divided into three groups:

| Group | % of data | Purpose |
|:---:|:---:|---|
| **Training** | 70% | The model learns from these |
| **Validation** | 15% | Used to check progress during training and tune settings |
| **Test** | 15% | Held aside, only used at the very end to measure final accuracy |

To make sure the model isn't just memorizing gene names, there's also a tougher test called **Gene-Held-Out** evaluation: entire genes are removed from training, so the model must generalize to genes it has **never seen**.

---

## The AI Model: What's Inside?

Don't worry about the math. Here's the intuition:

```
    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
    │  CHEMISTRY   │    │   CATEGORY   │    │   PROTEIN    │
    │   FACTS      │    │    LABELS    │    │   LANGUAGE   │
    │  (26 numbers)│    │  (type +     │    │   MODEL      │
    │              │    │   syndrome)  │    │  (ESM-2)     │
    └──────┬───────┘    └──────┬───────┘    └──────┬───────┘
           │                   │                   │
           └───────────────────┼───────────────────┘
                               │
                        ┌──────▼──────┐
                        │   COMBINE   │
                        │  all clues  │
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │   REFINE    │  ← Two "residual blocks" that
                        │   REFINE    │    process and improve the signal
                        └──────┬──────┘
                               │
                        ┌──────▼──────┐
                        │   DECIDE    │  → One number: P(dangerous)
                        └─────────────┘
```

**Think of it as a panel of three specialists:**
1. A **chemist** who measures how physically different the original and new amino acids are
2. A **geneticist** who knows what type of mutation this is and which cancer syndrome the gene is linked to  
3. A **protein expert** (ESM-2) who knows whether this amino acid "fits" at this protein position based on millions of known proteins

Their opinions are combined, refined twice, and distilled into a single danger score.

### Extra tricks to make it better

- **Ensemble of 5**: Instead of training one model, we train **5 copies** with different random starting points and average their opinions. Like asking 5 doctors instead of 1.
- **Calibration**: After training, we adjust the scores so they're accurate probabilities. If the model says "80% dangerous," it should actually be right about 80% of the time.
- **Early stopping**: Training stops automatically when the model starts getting worse on the validation set (to prevent memorization).

---

## How Accurate Is It?

| Metric | Score | What it means |
|---|:---:|---|
| **AUC** | 0.993 | The model correctly ranks a random dangerous variant above a random harmless one 99.3% of the time |
| **AUPRC** | 0.982 | High precision even when looking specifically for rare dangerous variants |
| **ECE (after calibration)** | 0.001 | The predicted probabilities are almost perfectly calibrated |

On the hardest test (**Gene-Held-Out**, where the model has never seen the gene before):
- Overall AUC: **0.992**
- Missense variant AUC: **0.886** (these are the trickiest type — one amino acid swapped for another)

---

## Files in This Project

```
Deep-Learning-project/
│
├── data/                          ← All the data
│   ├── clinvar_labelled_v3.csv    ← Labeled DNA typos from ClinVar
│   ├── v3_featured_snv.csv        ← Same data + computed chemistry features
│   ├── esm_cache.parquet          ← Cached protein language model results
│   ├── panel.json                 ← Info about the 130 cancer genes
│   ├── penetrance.csv             ← Published cancer risk numbers
│   ├── panel_proteins.fasta       ← Protein sequences for ESM-2
│   └── sample_demo.vcf            ← Example DNA file with 5 test variants
│
├── models/                        ← The trained AI
│   ├── genome_dl_fusion_v3.pth    ← Saved model weights (~7 MB)
│   └── ..._meta.json              ← Accuracy metrics
│
├── src/v3/                        ← The code
│   ├── stage0/                    ← Step 0: Prepare data
│   │   ├── extract_v3.py          ← Pull variants from ClinVar
│   │   ├── build_v3_features.py   ← Compute chemistry features
│   │   └── make_panel_json.py     ← Build gene info file
│   ├── esm_features.py            ← Step 1: Run protein language model
│   ├── train_fusion.py            ← Step 2: Train the AI model
│   ├── evaluate.py                ← Step 3: Test how accurate it is
│   ├── vcf_pipeline.py            ← Step 4: Analyze a patient's DNA
│   └── generate_pdf_report.py     ← Step 4: Make the PDF report
│
├── reports/                       ← Generated reports
├── tests/                         ← Automated tests
├── others/                        ← Archived older versions (v1, v2)
└── requirements.txt               ← Software dependencies
```

---

## How to Run It

### Install

```bash
cd Deep-Learning-project
pip install -r requirements.txt
```

### Analyze a DNA file (the main thing most people want to do)

```bash
python -m src.v3.vcf_pipeline data/sample_demo.vcf
```

This reads the demo VCF file and prints a clinical report. It also saves:
- `reports/vcf_report.txt` — text report
- `reports/vcf_report.json` — machine-readable results
- `reports/vcf_report.pdf` — formatted PDF report

### Train the model from scratch (optional — a pre-trained model is included)

```bash
# Step 1: Extract protein language model features (~30-60 min)
python -m src.v3.esm_features

# Step 2: Train the model (~10-30 min depending on GPU)
python -m src.v3.train_fusion

# Step 3: Run the full evaluation
python -m src.v3.evaluate
```

### Run tests

```bash
pytest tests/ -v
```

---

## Glossary (Plain English)

| Word | What it means |
|---|---|
| **DNA** | The molecule that stores your genetic instructions. Made of 4 "letters": A, C, G, T |
| **Gene** | A section of DNA that contains instructions for building one protein |
| **Protein** | A molecular machine in your body. Made of a chain of amino acids |
| **Amino acid** | Building blocks of proteins. There are 20 types. A DNA mutation can swap one for another |
| **Variant / Mutation / Typo** | A place where your DNA differs from the reference human genome |
| **SNV** | Single Nucleotide Variant — a single-letter DNA typo (e.g., A changed to G) |
| **Pathogenic** | Dangerous — this variant is known or predicted to cause disease |
| **Benign** | Harmless — this variant doesn't cause problems |
| **VUS** | Variant of Uncertain Significance — we don't know if it's dangerous or not |
| **VCF file** | The standard file format that stores DNA variant data from sequencing |
| **ClinVar** | A free public database where labs share their classifications of DNA variants |
| **GRCh38** | The current version of the "reference" human genome that everything is compared against |
| **ESM-2** | A protein language model by Meta AI — it has "read" millions of proteins and knows what looks normal vs. weird at each position |
| **Grantham score** | A number (0–215) that measures how chemically different two amino acids are |
| **Model** | The trained AI system that makes predictions |
| **Training** | The process of showing the AI thousands of examples so it learns patterns |
| **Epoch** | One complete pass through all the training data |
| **Batch** | A small group of examples processed together (default: 256 at a time) |
| **Loss** | A number measuring how wrong the model's predictions are. Training tries to make this smaller |
| **Overfitting** | When the model memorizes the training data instead of learning general patterns (bad!) |
| **Calibration** | Adjusting scores so "80% confident" really means "right about 80% of the time" |
| **Ensemble** | Training multiple models and averaging their predictions for better accuracy |
| **AUC** | A 0-to-1 score measuring how well the model separates dangerous from harmless variants. 1.0 = perfect |
| **Penetrance** | The chance that someone with a specific dangerous variant will actually develop cancer |

---

> ⚠️ **Reminder**: This is a research prototype. Any results must be confirmed by a certified clinical laboratory with genetic counselling before making medical decisions.
