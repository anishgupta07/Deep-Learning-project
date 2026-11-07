# GenomeDL v3: Comprehensive Architecture & Concepts Reference Guide

> **Author**: GenomeDL Core Engineering Team  
> **Document Type**: Architecture Specification, Biophysical Principles, & Complete Conceptual Glossary  
> **Scope**: Pediatric & Germline Hereditary Cancer Genomic Inference Pipeline (v3)

---

## Table of Contents
1. [Executive Overview & Clinical Mission](#1-executive-overview--clinical-mission)
2. [End-to-End System Pipeline Architecture](#2-end-to-end-system-pipeline-architecture)
3. [Deep Learning Model Architecture (`FusionResNet`)](#3-deep-learning-model-architecture-fusionresnet)
4. [Master Glossary: Exhaustive Explanation of Every Term](#4-master-glossary-exhaustive-explanation-of-every-term)
   - [A. Genomic Coordinates, Sequencing & VCF Terms](#a-genomic-coordinates-sequencing--vcf-terms)
   - [B. The 12 Genomic Variant Classes](#b-the-12-genomic-variant-classes)
   - [C. Physicochemical & Tabular Features (26 Inputs)](#c-physicochemical--tabular-features-26-inputs)
   - [D. Evolutionary Protein Language Modeling (ESM-2)](#d-evolutionary-protein-language-modeling-esm-2)
   - [E. Machine Learning, Training, Calibration & Evaluation](#e-machine-learning-training-calibration--evaluation)
   - [F. Clinical Genetics, Syndromes & Lifetime Penetrance](#f-clinical-genetics-syndromes--lifetime-penetrance)
5. [Codebase Organization & File Guide](#5-codebase-organization--file-guide)

---

## 1. Executive Overview & Clinical Mission

### The Diagnostic Challenge
Whole Genome Sequencing (WGS) and Whole Exome Sequencing (WES) of pediatric patients typically identify between **20,000 to 4,000,000 genetic variants** per individual. Within actionable cancer predisposition genes (e.g., *BRCA1*, *BRCA2*, *TP53*, *PALB2*, *MLH1*), clinicians must answer three critical questions rapidly:
1. **Which variants are harmless common polymorphisms?** (Must be eliminated to prevent false alarms).
2. **Which variants have proven, peer-reviewed clinical consensus?** (Must be retrieved directly from registries like ClinVar without guessing).
3. **What is the pathogenicity of novel, unclassified missense mutations?** (Where no clinical consensus exists, requiring molecular AI).
4. **What is the genuine lifetime cancer risk for the child or carrier?** (Must be derived from published epidemiological cohorts, never arbitrary mathematical formulas).

### The GenomeDL v3 Solution
GenomeDL v3 refactors ad-hoc prediction into an **8-stage clinical decision tree**:

```
Baby's VCF (.vcf / .vcf.gz)
   │
   ▼
[1. GRCh38 Assembly Check] ──(hg19 detected)──> Halt with liftOver instructions
   │
   ▼
[2. Quality Filtering] ──(Low DP < 10 or Low GQ < 20 or Filter != PASS)──> Dropped
   │
   ▼
[3. 130-Gene Panel Filter] ──(Outside targeted cancer regions)──> Excluded
   │
   ▼
[4. Common Variant Filter] ──(gnomAD Allele Frequency > 1.0%)──> BENIGN (Polymorphism)
   │
   ▼
[5. ClinVar Expert Lookup] ──(Review Stars ≥ 2)──> CLINVAR VERDICT (Zero model compute)
   │
   ▼
[6. Rule-Based LoF Logic] ──(Frameshift / Canonical Splice)──> PREDICTED LOSS-OF-FUNCTION
   │
   ▼
[7. Multimodal Fusion ResNet] ──(Remaining Unclassified SNVs)──> CALIBRATED P(pathogenic)
   │
   ▼
[8. Recessive Zygosity Check] ──(Single Het in MUTYH/NTHL1)──> RECESSIVE CARRIER (Not High Risk)
   │
   ▼
[9. Evidence-Based Risk Report] ──> Curated Published Penetrance Ranges (JAMA, NEJM)
                                  ──> Text (.txt), JSON (.json), & Clinical PDF (.pdf)
```

---

## 2. End-to-End System Pipeline Architecture

The pipeline processes input VCF files through eight sequential, reproducible stages:

### Stage 1: Reference Genome Assembly Verification
- **Mechanism**: Inspects the VCF header contig tags (`##contig=<ID=chr1,length=...>`) and `##reference`.
- **Logic**: Must match **GRCh38 / hg38**. If legacy `hg19`, `GRCh37`, or `b37` is detected, the pipeline halts immediately with an actionable error directing the user to lift over coordinates using NCBI Remap or UCSC liftOver.

### Stage 2: Quality & Biallelic Parsing
- **Filters**:
  - `FILTER == 'PASS'` or `'.'`. Flags like `LowQual`, `Artifact`, or `LowDepth` are excluded.
  - `DP >= 10`: Minimum 10 reads covering the locus.
  - `GQ >= 20`: Phred-scaled Genotype Quality $\ge 20$ (corresponding to $\le 1\%$ error probability in genotype assignment).
  - Multi-allelic records (e.g. `REF=A, ALT=C,T`) are decomposed into distinct biallelic records.

### Stage 3: Panel Coordinate Filtering
- **Panel Scope**: Exactly 130 hereditary cancer predisposition genes classified across 15 clinical syndrome groups (`data/panel.json` and `data/panel_coordinates.json`).
- **Optimization**: Coordinate interval lookups bypass non-target regions, allowing a 4-million-variant whole-genome VCF (like HG002) to be filtered down to candidate cancer variants in under 15 seconds.

### Stage 4: Population Frequency Filter (gnomAD)
- **Clinical Rule**: An allele frequency (AF) $> 1.0\%$ ($0.01$) in outbred continental populations is incompatible with high-penetrance, childhood- or early-onset dominant cancer syndromes.
- **Action**: Variants with $\text{AF} > 0.01$ are classified as **`BENIGN`** (Common Polymorphism). Deep learning inference is bypassed, ensuring zero false-positive alerts on common population variants.

### Stage 5: Curated Clinical Database Lookup (ClinVar 2+ Stars)
- **Lookup Key**: Genomic tuple `(chromosome, position, ref_allele, alt_allele)`.
- **Indexed Registry**: 148,747 ClinVar variants (`data/clinvar_labelled_v3.csv`).
- **Star Threshold**: If ClinVar review status has $\ge 2$ stars (reviewed by expert panel like ENIGMA, InSiGHT, or multiple concordant submitters), the ClinVar verdict is assigned directly as authoritative truth. Model computation is intentionally skipped.

### Stage 6: Rule-Based Loss-of-Function (LoF) Filter
- **Mechanism**: High-impact protein-truncating variants—specifically **frameshift indels**, **start lost**, and **canonical splice donor/acceptor mutations** (within $\pm 1, 2$ bp of exon boundaries)—in tumor suppressor genes are assigned **`Pathogenic (Predicted Loss-of-Function)`** via ACMG PVS1 rule logic, rather than being subjected to empirical continuous scoring.

### Stage 7: Multimodal Fusion ResNet Inference
- **Target Population**: Only the remaining **unclassified novel single nucleotide variants (SNVs)**—principally missense variants, non-canonical splice region variants, and UTR variants without expert consensus.
- **Inference Engine**: Evaluated via the `FusionResNet` incorporating 26 tabular biochemical features, categorical syndrome embeddings, and ESM-2 masked-marginal log-likelihood ratios and structural embeddings.
- **Output**: Calibrated probability $P(\text{pathogenic}) \in [0.0, 1.0]$.

### Stage 8: Zygosity & Recessive Inheritance Resolution
- **Genes**: Autosomal recessive cancer predisposition genes, specifically `MUTYH` (MUTYH-associated polyposis) and `NTHL1`.
- **Logic**: Heterozygous mutations (`0/1`) in recessive genes do not confer high-penetrance cancer risk. They are classified as **`Carrier (Recessive)`** with an explanatory clinical note, preventing inappropriate prophylactic surgical recommendations. Elevated cancer penetrance is only reported if biallelic (homozygous `1/1` or compound heterozygous with two distinct mutations in the same gene).

### Stage 9: Curated Lifetime Risk Reporting
- **Removed Anti-Pattern**: Naive multiplying formulas such as $P(\text{pathogenic}) \times \text{penetrance}$ were historically used in early prototypes, which produced scientifically unvalidated numbers.
- **Current Approach**: Published epidemiological lifetime carrier risk ranges are read directly from `data/penetrance.csv` with explicit source citations (*JAMA*, *NEJM*, *Lancet Oncology*, *NCCN Guidelines*). Non-cancer syndromic features (e.g. pneumothorax, leiomyomatosis) are separated into a dedicated list.

---

## 3. Deep Learning Model Architecture (`FusionResNet`)

The core deep learning model is a **Multimodal Fusion Residual Network (`FusionResNet`)** designed to evaluate protein biophysics and evolutionary fitness without memorizing gene labels.

```
Tabular Features (26 continuous & binary)
         │
         ▼
[Linear(26 -> 64) -> LayerNorm -> GELU]  ──────────────┐
                                                       │
Variant Class ID (0..11)                               │
         │                                             │
         ▼                                             │
[Embedding(12, 8)]  ───────────────────────────────────┼──> Concatenation (144-d)
                                                       │            │
Syndrome Group ID (0..14)                              │            ▼
         │                                             │    [Linear(144 -> 256)]
         ▼                                             │            │
[Embedding(15, 8)]  ───────────────────────────────────┤            ▼
                                                       │    [Residual Block 1 (256-d)]
ESM-2 Features                                         │    (LayerNorm -> GELU -> Dropout(0.2) -> Linear)
  ├── esm_llr (1-d)                                    │            │
  ├── esm_available (1-d)                              │            ▼
  └── esm_embedding (480-d -> Linear(480 -> 64)) ─────┘    [Residual Block 2 (256-d)]
                                                                    │
                                                                    ▼
                                                            [Linear(256 -> 1)]
                                                                    │
                                                                    ▼
                                                            Raw Logit z
                                                                    │
                                                                    ▼
                                                   Temperature Scaling: z / T
                                                                    │
                                                                    ▼
                                                   Calibrated P(pathogenic) = σ(z / T)
```

### Key Architectural Specifications:
1. **Gene Embedding OFF by Default**: In legacy v1/v2 models, embedding the gene symbol caused the network to memorize that *TP53* has many pathogenic mutations while *PMS2* has fewer, destroying out-of-gene generalization. In v3, gene embeddings are disabled (`use_gene_embedding=False`). The model evaluates biophysics, not gene identity.
2. **Residual Highway**: Two residual blocks with LayerNorm, GELU activations, and 20% Dropout prevent vanishing gradients and regularize representations across multi-seed training.
3. **Loss Function**: Weighted Binary Cross-Entropy with `pos_weight` set dynamically on training data:
   $$\text{pos\_weight} = \frac{N_{\text{benign}}}{N_{\text{pathogenic}}} \approx \frac{128{,}000}{20{,}700} \approx 6.18$$
4. **Post-Hoc Temperature Scaling Calibration**:
   Raw logits $z$ are calibrated via a learned temperature parameter $T > 0$ fitted on held-out validation data:
   $$\hat{P} = \sigma\left(\frac{z}{T}\right) = \frac{1}{1 + e^{-z / T}}$$
   On our trained checkpoint, $T = 0.9357$, reducing Expected Calibration Error (ECE) from $0.0276$ to $0.0249$ ($< 2.5\%$).

---

## 4. Master Glossary: Exhaustive Explanation of Every Term

### A. Genomic Coordinates, Sequencing & VCF Terms

#### DNA (Deoxyribonucleic Acid)
The hereditary chemical macromolecule consisting of two antiparallel polynucleotide chains coiled into a double helix. Built from four nitrogenous bases: Adenine (A), Cytosine (C), Guanine (G), and Thymine (T).

#### Chromosome
Dense structural packages of DNA and histone proteins. Humans normally have 23 pairs of chromosomes (22 autosomes numbered 1 through 22, and 1 pair of sex chromosomes, X and Y).

#### Locus (plural: Loci)
The precise, unique physical coordinate of a gene or DNA base on a specific chromosome (e.g., Chromosome 17, coordinate 43,106,487).

#### Reference Genome (GRCh38 / hg38)
The official digitized coordinate map of the human genome published by the Genome Reference Consortium (GRC). "GRCh38" (also called "hg38") is the modern reference standard released with updated centromeres, alternate loci, and corrected gene models. Legacy coordinates from "GRCh37" or "hg19" have different base positions (e.g., *BRCA1* is located around 41.2 Mb in hg19 but shifted to 43.1 Mb in GRCh38).

#### VCF (Variant Call Format)
The international standardized text file specification used in bioinformatics to record genetic differences relative to the reference genome. A VCF contains:
- **Header Lines (`##`)**: Definitions of reference genome, contig lengths, INFO tags, and FORMAT fields.
- **Column Header (`#CHROM POS ID REF ALT QUAL FILTER INFO FORMAT SAMPLE`)**: Identifies chromosome, coordinate, dbSNP identifier, reference base, alternate base, quality, filter flag, annotation fields, and sample genotypes.

#### REF (Reference Allele)
The nucleotide base(s) observed in the standard human reference genome at that exact position (e.g., `A`).

#### ALT (Alternate Allele)
The nucleotide base(s) observed in the patient/sequenced individual that differs from the reference (e.g., `C`).

#### SNV (Single Nucleotide Variant)
A variation in DNA sequence occurring when a single nucleotide ($A, C, G, \text{or } T$) in the genome differs from the reference genome. If present in $> 1\%$ of the general population, it is historically called a **SNP** (Single Nucleotide Polymorphism).

#### InDel (Insertion / Deletion)
A genetic variant where one or more nucleotides are added (insertion) or removed (deletion) from the DNA sequence.

#### Transition vs. Transversion (`is_transition`)
- **Transition**: A point mutation changing a purine into another purine ($A \leftrightarrow G$) or a pyrimidine into another pyrimidine ($C \leftrightarrow T$). Thermodynamically more common due to molecular shape similarity.
- **Transversion**: A point mutation changing a purine into a pyrimidine or vice versa ($A \leftrightarrow C$, $A \leftrightarrow T$, $G \leftrightarrow C$, $G \leftrightarrow T$). Mechanistically rarer and often structurally more disruptive.

#### Plus Strand vs. Coding Strand
- **Plus Strand (`+`)**: The standard $5' \to 3'$ coordinate orientation of the reference chromosome from pter to qter.
- **Minus Strand (`-`)**: The reverse-complement orientation. Many cancer genes (e.g., *BRCA1*, *TP53*, *PALB2*, *CHEK2*) are transcribed from the minus strand.
- **Coding Strand**: The actual DNA strand whose sequence corresponds to the mRNA sequence. For a minus-strand gene, an $A \to G$ change on the chromosome is actually a $T \to C$ change in the mRNA transcript. GenomeDL derives coding-strand one-hots (`cs_ref`, `cs_alt`) to ensure the neural net sees the true biological codon change.

#### Genotype (`GT`)
The specific pair of alleles possessed by an individual at a locus:
- `0/0`: Homozygous Reference (both copies match the reference genome).
- `0/1` or `0|1`: Heterozygous (one chromosome copy has REF, the other copy has ALT).
- `1/1` or `1|1`: Homozygous Alternate (both chromosome copies carry the mutation).

#### Read Depth (`DP`)
The total number of independent sequencing reads that mapped to and covered that exact genomic coordinate. Higher depth (e.g., $\text{DP} \ge 30$) provides higher statistical confidence that the variant is real biological DNA and not an optical or PCR sequencer artifact.

#### Genotype Quality (`GQ`)
A Phred-scaled confidence score representing the probability that the assigned genotype call is incorrect:
$$\text{GQ} = -10 \log_{10} P(\text{Genotype Call is Wrong})$$
A $\text{GQ} = 20$ corresponds to $99\%$ confidence ($1\%$ error rate); $\text{GQ} = 99$ indicates $> 99.99999\%$ certainty.

#### Allele Frequency (`AF`)
The proportion of chromosomes in a given population sample that carry the alternate allele. If an allele has $\text{AF} = 0.05$, $5\%$ of all tested chromosomes in that population carry that variation.

#### gnomAD (Genome Aggregation Database)
The world's largest reference database of human population variation, aggregating exome and genome sequences from $> 140{,}000$ unrelated individuals without severe pediatric disease. Serves as the primary negative control for human genetics.

#### ClinVar
A freely accessible, public archive hosted by the National Center for Biotechnology Information (NCBI) that documents human genetic variants and their relationship to disease phenotypes, supported by clinical consensus reviews.

#### ClinVar Review Status & Star System
- **0 Stars**: Variant submitted without criteria or conflicting interpretations.
- **1 Star**: Single submitter provided interpretation criteria.
- **2 Stars**: Multiple independent clinical submitters evaluated the variant with assertion criteria and reached concordant conclusions (No conflicts).
- **3 Stars**: Reviewed by an accredited expert clinical panel (e.g., ENIGMA for BRCA1/2, InSiGHT for Lynch Syndrome).
- **4 Stars**: Practice guideline asserted by professional societies (ACMG, AMP).

#### HGVS Nomenclature
The internationally accepted standard nomenclature maintained by the Human Genome Variation Society:
- `c.181T>G`: Coding DNA position 181 changed from Thymine to Guanine.
- `p.Cys61Gly` (or `p.C61G`): Protein amino acid residue 61 changed from Cysteine (Cys) to Glycine (Gly).
- `p.Ter214Leu` (or `p.*214L`): Stop codon at residue 214 mutated into Leucine, extending the protein (stop lost).

---

### B. The 12 Genomic Variant Classes

In Stage 0, the repository was standardized from inconsistent legacy labels into **12 distinct, mutually exclusive variant classes**:

| # | Variant Class Name | Molecular Consequence | Pathogenic Mechanism | Typical Pipeline Route |
| :---: | :--- | :--- | :--- | :--- |
| **1** | `missense` | Single amino acid substitution (e.g. Met $\to$ Val). | Disruption of active sites, folding destabilization, loss of binding interfaces. | Evaluated via Multimodal Fusion ResNet + ESM-2 |
| **2** | `nonsense` (`stop_gained`) | Introduces premature termination codon (PTC). | Truncates protein; triggers Nonsense-Mediated Decay (NMD) of mRNA. | Rule-based LoF (PVS1) |
| **3** | `synonymous` | Codon changes but encodes the identical amino acid. | Usually benign; rarely alters splicing enhancers or mRNA secondary structure. | Benign filter / scored with tabular branch |
| **4** | `frameshift` | Insertion or deletion not divisible by 3 bases. | Shifts open reading frame, generating completely aberrant peptide chain and early stop. | Rule-based LoF (PVS1) |
| **5** | `splice_donor` | Mutation within canonical +1 or +2 donor dinucleotide (GT). | Destroys $5'$ splice junction; leads to exon skipping or intron retention. | Rule-based LoF (PVS1) |
| **6** | `splice_acceptor` | Mutation within canonical -1 or -2 acceptor dinucleotide (AG). | Destroys $3'$ splice junction; leads to aberrant splicing. | Rule-based LoF (PVS1) |
| **7** | `splice_region` | Mutation in adjacent exon/intron boundary (e.g., -3 to +8 bp). | Weakens spliceosome recognition efficiency; partial exon skipping. | Evaluated via Multimodal Fusion ResNet |
| **8** | `start_lost` | Mutation in the initiation codon (AUG / Met1). | Abolishes translation initiation from canonical start site. | Fixed severity feature + ResNet scoring |
| **9** | `stop_lost` | Termination codon mutated to an amino acid residue. | Ribosome continues translating into $3'$ UTR, destabilizing protein. | Fixed severity feature + ResNet scoring |
| **10** | `utr5` (`5_prime_UTR`) | Mutation in $5'$ untranslated leader region of mRNA. | Can alter Ribosome Binding Site, Kozak consensus, or upstream Open Reading Frames (uORFs). | Scored via Tabular branch |
| **11** | `utr3` (`3_prime_UTR`) | Mutation in $3'$ untranslated trailer region of mRNA. | Can disrupt microRNA (miRNA) binding sites or polyadenylation signals. | Scored via Tabular branch |
| **12** | `intronic` | Deep non-coding mutation $> 20$ bp away from exon boundaries. | Vast majority benign; rare deep intronic variants create pseudo-exons. | Benign / Tabular branch scoring |

---

### C. Physicochemical & Tabular Features (26 Inputs)

The tabular branch processes 26 biochemically grounded continuous and binary features:

#### 1. Relative Protein Position (`rel_pos`)
$$\text{rel\_pos} = \frac{\text{aa\_pos}}{\text{protein\_length}}$$
Normalized position of the mutation along the peptide chain from $0.0$ ($N$-terminus) to $1.0$ ($C$-terminus). Crucial because mutations near the $N$-terminus or within folded functional domains often have different structural consequences than flexible $C$-terminal tails.

#### 2. Protein Length
The total number of amino acids in the primary canonical transcript. Stage 0 fixed this feature by generating `data/protein_lengths.json` directly from **MANE Select / UniProt**, eliminating legacy errors where length was estimated from max observed variant position.

#### 3. Grantham Distance Matrix (`grantham`)
A classic biophysical metric published by Richard Grantham (1974) measuring evolutionary distance between amino acid pairs based on three properties:
- Composition of atomic carbon chain ($c$)
- Polarity ($p$)
- Molecular volume ($v$)
Values range from $5$ (Leu $\to$ Ile, chemically similar) to $215$ (Cys $\to$ Trp, extreme structural contrast). Higher values correlate with higher probability of functional damage.

#### 4. Molecular Weight Delta ($\Delta MW$)
The absolute difference in mass (in Daltons / $\text{g/mol}$) between the reference and alternate amino acid side chains:
$$\Delta MW = |MW_{\text{ref}} - MW_{\text{alt}}|$$
For example, Glycine ($MW \approx 75.0$) substituted by Tryptophan ($MW \approx 204.2$) introduces a massive $\Delta MW = 129.2$, often disrupting steric packing inside the protein core.

#### 5. Net Isoelectric Charge Delta ($\Delta Charge$)
The change in net electrical formal charge at physiological pH ($7.4$):
$$\Delta Charge = |Charge_{\text{ref}} - Charge_{\text{alt}}|$$
Lysine ($+1$) or Arginine ($+1$) mutated to Aspartate ($-1$) yields $\Delta Charge = 2.0$, which destroys electrostatic salt bridges.

#### 6. Lipophilicity / Hydrophobicity Delta ($\Delta \log P$)
The difference in octanol-water partition coefficient ($\log P$), which quantifies how hydrophobic or hydrophilic an amino acid side chain is. A hydrophobic-to-polar mutation inside the buried hydrophobic core causes energetic destabilization and unfolding.

#### 7. Topological Polar Surface Area Delta ($\Delta TPSA$)
The difference in the surface sum of all polar atoms (oxygen, nitrogen, and attached hydrogens) in $\text{\AA}^2$. Reflects the hydrogen-bonding capacity of the substituted residue.

#### 8. Morgan Fingerprint Tanimoto Dissimilarity (`1 - Tanimoto`)
Using RDKit cheminformatics, each 20 canonical amino acid is represented as a circular Morgan chemical fingerprint. The Tanimoto similarity coefficient measures the Jaccard overlap of chemical subgraphs:
$$\text{Tanimoto}(A, B) = \frac{|A \cap B|}{|A \cup B|}$$
The feature used is the chemical dissimilarity $1 - \text{Tanimoto} \in [0.0, 1.0]$.

#### 9. Base One-Hot Encodings
Binary flags indicating the exact nucleotide identity on both the genomic plus strand (`ref_A`, `ref_C`, `ref_G`, `ref_T`, `alt_A`, `alt_C`, `alt_G`, `alt_T`) and the biological coding strand (`cs_ref_*`, `cs_alt_*`).

---

### D. Evolutionary Protein Language Modeling (ESM-2)

#### ESM-2 (Evolutionary Scale Modeling)
A deep Transformer protein language model developed by Meta AI (Fair). Trained on hundreds of millions of diverse evolutionary protein sequences from UniRef50. ESM-2 learns the complex syntax, secondary structure, tertiary contacts, and functional constraints of proteins without supervision.

#### Architecture: `facebook/esm2_t12_35M_UR50D`
The default model employed in GenomeDL v3:
- 12 Transformer layers
- 35 Million parameters
- 480-dimensional hidden states
- Fast execution on consumer GPUs (RTX 3050, 4GB VRAM) and CPU inference in $< 50$ ms.

#### Masked Language Modeling (MLM)
During pretraining, tokens in the protein sequence are randomly masked (`<mask>`), and the model predicts which amino acid belongs at that position based on the contextual evolutionary grammar of the surrounding sequence.

#### Masked-Marginal Log-Likelihood Ratio (ESM-LLR)
The mathematical metric used by GenomeDL v3 to quantify the evolutionary penalty of a missense mutation.
The variant position $i$ in the protein sequence $S$ is masked ($S_{\backslash i} = S \text{ with position } i \text{ masked}$). In a single forward pass, the model outputs log-probabilities for all 20 standard amino acids:
$$\text{ESM\_LLR} = \log P(\text{alt\_aa} \mid S_{\backslash i}) - \log P(\text{ref\_aa} \mid S_{\backslash i})$$
- If $\text{ESM\_LLR} < 0$: The alternate amino acid is evolutionarily disfavored compared to the wild-type residue. Highly negative values (e.g. $-11.5$) strongly indicate pathogenic disruption.
- If $\text{ESM\_LLR} \ge 0$: The alternate amino acid is evolutionarily neutral or favored (characteristic of benign polymorphisms).

#### Centered Context Windowing
ESM-2 has a maximum context length of 1022 tokens. However, essential cancer proteins are much longer:
- **BRCA2**: 3,418 amino acids
- **ATM**: 3,056 amino acids
If $\text{protein\_length} > 1020$, GenomeDL v3 extracts a centered sliding window of length 1020 centered at the variant position:
$$\text{Window} = \left[\max\left(0, \text{pos} - 510\right), \min\left(L, \text{pos} + 510\right)\right]$$
This guarantees that long proteins never trigger out-of-memory errors or context clipping.

#### Masked-Position Hidden Embedding
In addition to the scalar LLR, the model extracts the 480-dimensional hidden state vector $\mathbf{h}_i \in \mathbb{R}^{480}$ from layer 12 at the masked variant position. This vector captures the local 3D structural micro-environment (backbone torsion angles, solvent accessibility, electrostatic contacts) and is projected into a 64-dimensional dense representation inside `FusionResNet`.

---

### E. Machine Learning, Training, Calibration & Evaluation

#### ResNet (Residual Network) & Skip Connections
A deep neural network architecture that utilizes additive shortcut connections:
$$\mathbf{y} = \mathbf{x} + \mathcal{F}(\mathbf{x})$$
Allows gradients to propagate unimpeded through dozens of layers, preventing gradient degradation during backpropagation.

#### Layer Normalization (`LayerNorm`)
Unlike Batch Normalization (which computes statistics across batches and behaves inconsistently with small batch sizes or variable inputs), LayerNorm normalizes across the feature dimensions of each individual sample independently:
$$\text{LN}(\mathbf{x}) = \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \odot \gamma + \beta$$
Stabilizes training dynamics in deep tabular and multimodal fusion architectures.

#### GELU (Gaussian Error Linear Unit)
A smooth, non-linear activation function defined as:
$$\text{GELU}(x) = x \cdot \Phi(x) = x \cdot P(X \le x), \quad X \sim \mathcal{N}(0, 1)$$
Unlike standard ReLU (which has a hard non-differentiable corner at 0), GELU weights inputs by their probability under a Gaussian distribution, improving gradient flow and model convergence in Transformer and ResNet heads.

#### Class Imbalance & Weighted BCE
In genomic datasets, harmless benign variants outnumber pathogenic cancer mutations by roughly $6:1$ to $7:1$. Standard Binary Cross-Entropy would cause the model to achieve $86\%$ accuracy simply by predicting zero for every sample. We employ Weighted Binary Cross-Entropy with positive weight $w$:
$$\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \left[ w \cdot y_i \log(\hat{p}_i) + (1 - y_i) \log(1 - \hat{p}_i) \right]$$

#### Post-Hoc Temperature Scaling
A simple, effective method for calibrating neural network probabilities without altering classification accuracy. Given unnormalized logit $z$, temperature $T > 0$ rescales the confidence:
$$\hat{p} = \sigma\left(\frac{z}{T}\right)$$
- If $T > 1$: Softens overconfident probabilities toward $0.5$.
- If $T < 1$: Sharpens underconfident probabilities toward $0.0$ or $1.0$.

#### ECE (Expected Calibration Error)
Measures the difference between predicted confidence and true empirical accuracy by partitioning predictions into $M$ bins (typically 10 or 15):
$$\text{ECE} = \sum_{m=1}^M \frac{|B_m|}{N} \left| \text{acc}(B_m) - \text{conf}(B_m) \right|$$
An $\text{ECE} = 0.0249$ means that across all risk bins, predicted probabilities deviate from empirical truth by only $2.49\%$.

#### ROC-AUC (Receiver Operating Characteristic - Area Under Curve)
The integral area under the curve plotting True Positive Rate vs False Positive Rate across all classification thresholds. Statistically equivalent to the Mann-Whitney $U$ statistic: the probability that a randomly chosen pathogenic variant will be assigned a higher risk score than a randomly chosen benign variant.

#### AUPRC (Area Under Precision-Recall Curve)
The integral area under the curve plotting Precision ($\frac{TP}{TP+FP}$) against Recall ($\frac{TP}{TP+FN}$). The gold standard metric for severe class imbalance. A random classifier scores equal to the positive prevalence ($\approx 0.14$). High AUPRC ensures high diagnostic precision without excessive false alarms.

#### 95% Bootstrap Confidence Intervals
To avoid cherry-picking single-split numbers, metrics are evaluated across 100 independent bootstrap resamples of the test set. The $2.5^{\text{th}}$ and $97.5^{\text{th}}$ percentiles define the $95\%$ Confidence Interval $[CI_{\text{low}}, CI_{\text{high}}]$. If two models have overlapping CIs, no claim of superiority is made.

#### Shuffled-Label Sanity Test
A rigorous negative control where true labels are randomly shuffled across the dataset. The model's evaluated ROC-AUC must collapse to $[0.45, 0.55]$ (centered at $0.50$). Any score $> 0.55$ proves the existence of target leakage, pipeline bugs, or data contamination.

---

### F. Clinical Genetics, Syndromes & Lifetime Penetrance

#### ACMG/AMP 5-Tier Classification System
The consensus framework established by the American College of Medical Genetics and Genomics and the Association for Molecular Pathology:
1. **Pathogenic**: Proven disease-causing variant supported by strong clinical, functional, or genetic evidence ($> 99\%$ certainty).
2. **Likely Pathogenic**: High probability of disease causation ($> 90\%$ certainty).
3. **Variant of Uncertain Significance (VUS)**: Insufficient or conflicting evidence; cannot be used for clinical decision-making.
4. **Likely Benign**: High probability of being harmless ($> 90\%$ certainty).
5. **Benign**: Harmless common polymorphism ($> 99\%$ certainty).

#### Carrier Status (Autosomal Dominant vs. Autosomal Recessive)
- **Autosomal Dominant** (e.g. *BRCA1*, *TP53*, *MLH1*): Carrying a single mutant allele (`0/1`) is sufficient to confer high lifetime cancer risk.
- **Autosomal Recessive** (e.g. *MUTYH*, *NTHL1*): Both alleles must be defective (`1/1` or compound heterozygous) to cause high-penetrance polyposis/colorectal cancer. A single heterozygous carrier (`0/1`) does not have high-penetrance risk and must not be given misleading cancer warnings.

#### Lifetime Penetrance
The cumulative probability that a person carrying a specific genetic mutation will develop the clinical disease by a specified age horizon (e.g., up to age 70 or 80).
- Incomplete Penetrance: Not all individuals carrying the mutation will develop cancer (e.g., *BRCA1* breast cancer penetrance is $65\% - 72\%$, meaning $\approx 30\%$ of carriers do not develop breast cancer).

#### Population Baseline Risk
The background lifetime cancer incidence in the general population, regardless of single-gene mutations (sourced from SEER / Cancer Research UK):
- General Population Breast Cancer: $\approx 12.9\%$
- General Population Ovarian Cancer: $\approx 1.3\%$
- General Population Colorectal Cancer: $\approx 4.1\%$

#### Curated Syndrome Profiles in GenomeDL v3:

| Syndrome Group | Exemplar Genes | Associated Cancers | Published Penetrance Range | Key Citations |
| :--- | :--- | :--- | :---: | :--- |
| **Hereditary Breast & Ovarian Cancer (HBOC)** | `BRCA1`, `BRCA2`, `PALB2`, `ATM`, `CHEK2` | Breast, Ovarian, Prostate, Pancreatic | Breast: 65%–72% (BRCA1), 61%–69% (BRCA2), 40%–60% (PALB2) | Kuchenbaecker et al. *JAMA* 2017; Antoniou et al. *NEJM* 2014 |
| **Li-Fraumeni Syndrome** | `TP53` | Soft tissue sarcoma, Osteosarcoma, Breast, Brain, Adrenocortical | All cancers combined: 70%–90% by age 70 | Chompret et al.; Olivier et al. 2010; NCCN 2024 |
| **Lynch Syndrome (HNPCC)** | `MLH1`, `MSH2`, `MSH6`, `PMS2` | Colorectal, Endometrial, Ovarian, Gastric | Colorectal: 40%–60% (MLH1/MSH2), 15%–22% (MSH6) | Dominguez-Valentin et al. *Lancet Oncol* 2020 |
| **Familial Adenomatous Polyposis (FAP)** | `APC` | Colorectal (thousands of adenomatous polyps), Duodenal | Colorectal: 95%–100% by age 50 | NCCN 2024 Guidelines |
| **Cowden Syndrome** | `PTEN` | Breast, Follicular Thyroid, Endometrial, Hamartomas | Breast: 67%–85%, Thyroid: 20%–35% | Tan et al. *Clin Cancer Res* 2012 |
| **Hereditary Diffuse Gastric Cancer** | `CDH1` | Diffuse gastric cancer, Lobular breast cancer | Gastric: 56%–70%, Lobular Breast: 39%–52% | Hansford et al. *Lancet Oncol* 2015 |
| **Multiple Endocrine Neoplasia** | `RET`, `MEN1` | Medullary thyroid carcinoma, Pheochromocytoma, Parathyroid | Medullary Thyroid: 90%–100% | Wells et al. *Thyroid* 2015 |
| **Von Hippel-Lindau (VHL)** | `VHL` | Clear cell renal carcinoma, Hemangioblastoma, Pheochromocytoma | Renal: 40%–70%, Pheo: 10%–20% | Maher et al. *Eur J Hum Genet* 2011 |

#### Non-Cancer Syndromic Features
Certain cancer predisposition genes cause prominent non-neoplastic physical symptoms that must be distinguished from cancer:
- `FLCN`: Spontaneous pneumothorax (collapsed lung) & pulmonary cysts (Birt-Hogg-Dubé syndrome).
- `FH`: Cutaneous and uterine leiomyomatosis (HLRCC syndrome).
- `ANKRD26`: Inherited thrombocytopenia (low platelet counts).
- `DKC1`: Aplastic anemia and bone marrow failure (Dyskeratosis congenita).

#### Mandatory Clinical Disclaimer
*Every automated report printed by GenomeDL v3 mandates the clinical alert*:
> **"Research prototype, not a diagnosis. Confirm in a clinical lab with genetic counselling. Testing children for adult-onset conditions needs counselling."**

---

## 5. Codebase Organization & File Guide

```
d:/Dl_project/
├── data/
│   ├── panel.json                     # 130 genes: syndrome group, strand, protein length, transcript
│   ├── panel_coordinates.json         # GRCh38 genomic coordinates (chrom, start, end) for all 130 genes
│   ├── panel_proteins.fasta           # Canonical amino acid sequences for all 130 panel genes
│   ├── penetrance.csv                 # Curated published lifetime risk estimates with citations
│   ├── clinvar_labelled_v3.csv        # Cleaned dataset of 148,747 ClinVar variants with 12 classes
│   ├── esm_cache.parquet              # Precomputed ESM-2 masked marginal LLRs and 480-d hidden states
│   ├── esm_embeddings.pt              # Full tensor of cached ESM embeddings
│   ├── sample_demo.vcf                # Synthetic demonstration VCF with known positive and negative controls
│   └── hg002_cancer_panel.vcf.gz      # Sliced 11,890 panel variants from NIST HG002 benchmark genome
├── models/
│   ├── genome_dl_fusion_v3.pth        # Trained Multimodal Fusion ResNet checkpoint (5-seed weights)
│   └── genome_dl_fusion_v3_meta.json  # Checkpoint metadata (temperature T=0.9357, validation metrics)
├── reports/
│   ├── eval.md                        # Master evaluation benchmark report across 3 splits
│   ├── vcf_report.txt                 # Human-readable text clinical report
│   ├── vcf_report.json                # Machine-readable structured JSON clinical report
│   ├── vcf_report.pdf                 # Publication-grade 2-page hospital clinical genetics PDF report
│   └── hg002_report.pdf               # Clinical PDF report for HG002 negative control benchmark
├── src/
│   ├── v3/
│   │   ├── stage0/                    # Stage 0 scripts: cleaning, panel construction, feature parity
│   │   ├── fetch_proteins.py          # Downloads UniProt canonical sequences for all 130 genes
│   │   ├── esm_features.py            # ESM-2 masked marginal extraction and parquet caching
│   │   ├── train_fusion.py            # Multimodal Fusion ResNet training and temperature scaling
│   │   ├── evaluate.py                # 3-split honest evaluation framework with bootstrap CIs
│   │   ├── generate_pdf_report.py     # Hospital-grade 2-page clinical PDF report generator
│   │   └── vcf_pipeline.py            # End-to-end VCF clinical inference pipeline
├── tests/
│   ├── test_stage0.py                 # Tests for Stage 0 fixes (10 tests)
│   ├── test_esm.py                    # Tests for ESM-2 feature parity and sequence matching (8 tests)
│   ├── test_eval.py                   # Tests for evaluation report and sanity checks (3 tests)
│   ├── test_vcf_pipeline.py           # Tests for VCF parser, ClinVar lookup, rules, disclaimer (6 tests)
│   └── test_pdf_report.py             # Tests for 2-page clinical PDF report generation (1 test)
└── docs/
    ├── ARCHITECTURE_AND_CONCEPTS.md   # THIS COMPREHENSIVE ARCHITECTURAL & GLOSSARY GUIDE
    ├── README_V3.md                   # Quickstart operational instructions
    └── Project Explainer.md           # High-level overview
```

---

*End of Architecture & Concepts Reference Guide.*
