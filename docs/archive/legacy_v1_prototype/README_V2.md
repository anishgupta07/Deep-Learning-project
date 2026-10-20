# GenomeDL — Hereditary-Cancer Variant Pathogenicity: Verified README (v2)

> **Purpose of this file.** A single, evidence-backed description of the project: what changed between the original repo (**v1**) and the rebuilt data pipeline (**v2**), what every dataset contains, the full model architecture, and every claim that was verified or refuted. It is written so that you can hand it to *any* person or AI model and they can **re-check each statement** with the commands in §12.
>
> **Status in one paragraph.** The **v2 dataset is verified correct** (it reproduces the raw-file extraction exactly, matches its manifest, and agrees 100% on labels with v1 where they overlap). The **v1 model code and checkpoint are real and reproducible** (test AUC 0.9652, sensitivity 92.35% reproduce exactly), but many v1 README claims are **false or unsupported** (§9), and the v1 evaluation is inflated by easy variants and by neighbour leakage (§8). **The model has not been retrained on v2**, and the v2 CSVs **do not yet contain engineered features** (§6). This is a research prototype, **not** a clinical tool.

**Conventions used below.** ✅ verified by me against files/code/data · ❌ verified false · ⚠️ partly true / misleading · ❓ not verifiable in my environment (reason given).

---

## Table of contents
1. Project goal and scope
2. Version history — what changed from v1 to v2
3. Data source and provenance
4. The v2 dataset (files, schema, counts)
5. The 130-gene panel
6. Feature engineering (v1 code, its defects, v2 status)
7. Model architecture and training (v1, fully specified)
8. Evaluation findings (measured numbers)
9. Audit of the original v1 README claims
10. What I could NOT verify
11. Recommended next steps and acceptance criteria
12. How to verify everything yourself
13. Prompt template for another model
14. Glossary

---

## 1. Project goal and scope

- **Task:** given a single-nucleotide variant (SNV) in a hereditary-cancer-predisposition gene, predict whether ClinVar classifies it as **pathogenic/likely pathogenic (label 1)** or **benign/likely benign (label 0)**. Binary classification.
- **Out of scope / not achieved:** clinical diagnosis, ACMG/AMP-compliant classification, personalised lifetime cancer risk, VCF ingestion, generalisation to unseen genes.
- **Honest framing:** ClinVar labels are themselves produced by curators using rules (e.g. "null variant ⇒ pathogenic", "synonymous ⇒ benign", population frequency, computational predictors). A model trained on them partly **re-learns curator rules** (circularity). Results are therefore "agreement with ClinVar", not ground-truth biology.

---

## 2. Version history — what changed from v1 to v2

### 2.1 Side-by-side

| Aspect | v1 (original zip `Deep-Learning-project-ftp-dataset`) | v2 (current, this README) |
|---|---|---|
| Extraction script | `build_pan_cancer_germline_dataset.py` (needs RDKit, also computes features) | `extract_clinvar_hereditary.py` (stdlib only, streams gz, **no features**) |
| Genes | **93** genes (README says 80), incl. somatic drivers (KRAS, EGFR, BRAF, …) | **130** hereditary-predisposition genes in 15 groups |
| Gene overlap | — | 69 genes shared with v1; 61 new; 24 v1 genes dropped |
| Review-status filter | none (ReviewStatus ignored) | `--min-stars 1` default (labelled benign/pathogenic only) |
| Labelled SNVs | 119,045 (100,997 benign / 18,048 pathogenic) | **148,825** (127,657 / 21,168) |
| VUS SNVs | 135,839 | **175,796** |
| Conflicting SNVs | 32,656 (file shipped, **no script produced it**) | **36,608** (produced by script) |
| Indels / MNVs | 49,105 (file shipped, **no script produced it**) | **40,153** labelled (produced by script, `--indels`) |
| Variant class column | none (derived ad hoc by me from flags) | `variant_class` (11 values, §4.3) |
| Protein-less variants | 34.8% of rows had no protein info → all-zero features + sentinel position 0.5 | explicit classes: `intronic`, `splice_canonical`, `splice_region`, `utr5`, `utr3`, `other_noncoding`, … |
| Start-loss | n/a | `start_lost` (fixed in a second pass, §2.3) |
| Transcript handling | none (mixed transcripts) | `transcript` + `on_primary_transcript` columns |
| `ref == alt` rows | present (HGVS `c.123=` entries) | removed (41 across all files, counted in manifest) |
| Provenance | none | `manifest.json` (params, input size/date, counts, per-gene table, primary transcripts) |
| Protein lengths | hand-typed dict (152 rows had relative position > 1) | not used; raw `aa_pos` only |
| Engineered features (Grantham, ΔMW, …) | in CSV | **not in v2 CSV — must be rebuilt** (§6) |
| Model | ResNet, checkpoint shipped | **unchanged, not retrained on v2** |

### 2.2 Change log (chronological)

1. **Audit of v1** (code + data + checkpoint) → found the defects in §9 and §8.
2. **Raw-file audit** (`audit_variant_summary.py`, run by you on the full ClinVar file): v1 extraction reproduced **exactly** (100,997 / 18,048 / 135,839). Details in §3.2.
3. **v2 extractor written** (130 genes, review stars, variant classes, manifest).
4. **v2 run #1 reviewed** → three defects found (my classifier bug and two data issues):
   - `p.Met1Ile` / `p.Met1Val` start-loss variants were classed `missense` (196 rows, all pathogenic → free shortcut "position 1 ⇒ pathogenic").
   - 30 labelled rows with `ref == alt` (HGVS `c.123=`).
   - 33 genes had >1 transcript in `Name`; 125 codons (383 rows) had conflicting reference amino acids.
5. **Extractor patched and v2 run #2 verified** (this README's dataset): all three fixed (§2.3).

### 2.3 Run #1 → run #2 verified diff

| Check | Run #1 | Run #2 (current) |
|---|---|---|
| Rows with ref == alt (labelled) | 30 | **0** |
| `missense` rows at `aa_pos == 1` | 196 | **0** (all 196 now `start_lost`, 100% pathogenic) |
| Codons with conflicting ref amino acid | 125 of 66,617 | **0 of 66,516** (primary transcript only) |
| Missense rows / % pathogenic | 9,955 / 42.2% | **9,759 / 41.1%** |
| Labelled benign | 127,687 | 127,657 (−30) |
| Labelled pathogenic | 21,168 | 21,168 (±0) |
| VUS / conflicting / indels | 175,799 / 36,611 / 40,154 | 175,796 / 36,608 / 40,153 |
| Variation-ids removed / added | — | 30 / 0 |
| Label changes on shared ids | — | 0 |
| Class changes on shared ids | — | 196 (all missense → start_lost) |

---

## 3. Data source and provenance

### 3.1 Source file
- **ClinVar `variant_summary.txt.gz`** (NCBI FTP, `tab_delimited/`).
- Input recorded in `manifest.json`: file `variant_summary.txt.gz`, **450,727,547 bytes**, modification time **2026-10-05** (local file mtime — *not* necessarily the ClinVar release date; record the real release/download date yourself).
- Total data rows: **9,223,698**. v1 and v2 runs used the same input (same size, same row count).
- Columns the extractor requires: `GeneSymbol, Assembly, OriginSimple, ClinicalSignificance, ReviewStatus, Type, Name, Chromosome, PositionVCF, ReferenceAlleleVCF, AlternateAlleleVCF`; optional: `VariationID, LastEvaluated`. The script exits with the actual header if any required column is missing.

### 3.2 Raw-file audit of v1 extraction (output you ran, `audit_variant_summary.py`)

| Step | Count |
|---|---|
| Total rows | 9,223,698 |
| Rows in the 93 v1 genes | 709,707 |
| GRCh38 | 352,485 |
| Germline | 335,668 |
| Not single-nucleotide | 48,249 |
| SNV | 287,419 |
| Conflicting / other SNV | 32,535 |
| Benign (0) | **100,997** |
| Pathogenic (1) | **18,048** |
| VUS (2) | **135,839** |

Identity check: 287,419 − 32,535 = 254,884 = 100,997 + 18,048 + 135,839 ✅. These equal v1's README/CSV numbers to the digit ⇒ **v1 extraction logic was correct**; v1's problems are design/evaluation, not extraction.

**Review status of v1 labels (SNV):**

| Review status | Benign | Pathogenic |
|---|---|---|
| criteria provided, single submitter | 52,950 (52.4%) | 6,480 (35.9%) |
| criteria provided, multiple submitters, no conflicts | 44,121 (43.7%) | 8,936 (49.5%) |
| reviewed by expert panel | 3,030 (3.0%) | 2,272 (12.6%) |
| no assertion criteria provided | 896 (0.9%) | 360 (2.0%) |

Only 1.1% (1,256) of v1 labels had no assertion criteria, so the star filter in v2 is a modest robustness measure, not a major cleaning step. Parsable protein change in `Name`: benign 64,980 / 100,997; pathogenic 12,688 / 18,048 (→ 5,360 pathogenic = 29.7% had no protein change).

### 3.3 Label rule (identical in v1 and v2; verified by exact count reproduction)
Lower-cased `ClinicalSignificance`:
- **1 (pathogenic):** contains "pathogenic" and none of {conflict, uncertain, benign}
- **0 (benign):** contains "benign" and none of {conflict, uncertain, pathogenic}
- **2 (VUS):** contains "uncertain" and not "conflict" (v2 also requires no pathogenic/benign)
- **−1 (conflicting/other):** everything else
- Included: `Assembly == GRCh38`, `"germline" in OriginSimple`, SNV = `len(ref)==len(alt)==1` using `ReferenceAlleleVCF/AlternateAlleleVCF`.
- Not included in labels: "risk factor", "association", "affects", "drug response" alone, etc.

---

## 4. The v2 dataset

### 4.1 Files (produced by `extract_clinvar_hereditary.py`, md5 `aff364aab833f1b882b66ecd64e3e751`)

| File | Rows | Bytes | Use |
|---|---|---|---|
| `clinvar_labelled_snv_1.csv` | **148,825** | 30,174,103 | training / evaluation (labels 0/1) |
| `clinvar_vus_snv_1.csv` | 175,796 | 37,543,164 | **scoring only — never train or compute headline metrics** (label = 2) |
| `clinvar_conflicting_snv_1.csv` | 36,608 | 8,703,898 | excluded from training (label = −1) |
| `clinvar_labelled_indels_1.csv` | 40,153 | 8,271,832 | labelled small indels and multi-base substitutions ≤ 50 bp (optional second task) |
| `manifest_1.json` | — | 24,948 | parameters, counts, per-gene table, primary transcripts |

(The `_1` suffix is only the upload name; the extractor writes them without it.)

### 4.2 Schema (all four CSVs share it)
`variation_id, gene, syndrome_group, chrom, pos, ref, alt, label, clinical_significance, review_status, stars, variant_class, ref_aa, alt_aa, aa_pos, transcript, hgvs_name, last_evaluated, on_primary_transcript`

| Column | Meaning |
|---|---|
| `variation_id` | ClinVar VariationID |
| `gene`, `syndrome_group` | panel gene and its group (§5) |
| `chrom`, `pos`, `ref`, `alt` | GRCh38 VCF coordinates (**plus strand**) |
| `label` | 1 pathogenic, 0 benign, 2 VUS, −1 conflicting |
| `clinical_significance`, `review_status`, `stars` | raw ClinVar text; stars mapped as below |
| `variant_class` | molecular class parsed from HGVS `Name` (§4.3) |
| `ref_aa`, `alt_aa`, `aa_pos` | from `(p.Xxx123Yyy)`; `.`/`0` when absent; stop = `*` |
| `transcript` | NM_/NR_ accession (with version) parsed from `Name` |
| `on_primary_transcript` | 1 if the row's transcript is the gene's most frequent transcript in the labelled+VUS files |
| `hgvs_name`, `last_evaluated` | raw ClinVar fields |

**Stars mapping:** practice guideline 4 · reviewed by expert panel 3 · criteria provided, multiple submitters, no conflicts 2 · criteria provided, single submitter 1 · criteria provided, conflicting classifications/interpretations 1 · everything else (incl. "no assertion criteria provided") 0. `--min-stars` is applied **only to labelled benign/pathogenic rows**; VUS keep all stars (VUS star counts: 0★ 783, 1★ 105,933, 2★ 67,953, 3★ 1,127).

### 4.3 `variant_class` rules (parsed from `Name`)
| Class | Rule |
|---|---|
| `synonymous` | `(p.Xxx123=)` |
| `nonsense` | `(p.Xxx123Ter)` or `*` |
| `start_lost` | `p.Met1?` **or** `p.Met1<any aa>` (ref Met, pos 1) |
| `missense` | `(p.Xxx123Yyy)` otherwise |
| `frameshift` | `(p.Xxx123…fs)` (indel file) |
| `splice_canonical` | no protein change, c. offset `±1…2` |
| `splice_region` | c. offset `±3…8` |
| `intronic` | c. offset beyond ±8 |
| `utr5` / `utr3` | `c.-N` / `c.*N` |
| `coding_no_protein_annot` | c. position without offset or protein change |
| `other_noncoding` | anything else (e.g. `n.`) |

Caveats: `splice_region` / `intronic` are by HGVS offset only (no splice-effect prediction); an exonic variant near an exon boundary stays `missense`/`synonymous`.

### 4.4 Labelled SNV composition (v2, current)
| variant_class | benign (0) | pathogenic (1) | total | % pathogenic | % of all rows |
|---|---|---|---|---|---|
| synonymous | 78,344 | 97 | 78,441 | 0.1 | 52.7 |
| intronic | 32,976 | 296 | 33,272 | 0.9 | 22.4 |
| nonsense | 11 | 10,444 | 10,455 | 99.9 | 7.0 |
| missense | 5,751 | 4,008 | 9,759 | 41.1 | 6.6 |
| splice_region | 7,451 | 458 | 7,909 | 5.8 | 5.3 |
| splice_canonical | 3 | 5,603 | 5,606 | 99.9 | 3.8 |
| utr3 | 1,459 | 6 | 1,465 | 0.4 | 1.0 |
| utr5 | 1,344 | 56 | 1,400 | 4.0 | 0.9 |
| other_noncoding | 315 | 4 | 319 | 1.3 | 0.2 |
| start_lost | 0 | 196 | 196 | 100.0 | 0.1 |
| coding_no_protein_annot | 3 | 0 | 3 | 0.0 | 0.0 |

**Reading this table correctly:** 75% of rows are `synonymous` + `intronic` (almost all benign); `nonsense`, `splice_canonical`, `start_lost` are ≥99.9% pathogenic and therefore trivial. The **clinically hard** classes are **missense (9,759 rows, 41.1% pathogenic)** and **splice_region (7,909 rows, 5.8%)**. Any single headline metric over all rows is dominated by easy rows.

**v1 composition for comparison (derived from v1 CSV flags):**
| variant class (v1 flags) | benign (0) | pathogenic (1) | total | % pathogenic | % of all rows |
|---|---|---|---|---|---|
| synonymous | 60,027 | 95 | 60,122 | 0.2 | 50.5 |
| no_protein_annot | 36,017 | 5,360 | 41,377 | 13.0 | 34.8 |
| missense | 4,884 | 3,987 | 8,871 | 44.9 | 7.5 |
| nonsense | 69 | 8,585 | 8,654 | 99.2 | 7.3 |
| other | 0 | 21 | 21 | 100.0 | 0.0 |

### 4.5 Review stars (labelled SNVs, v2)
| stars | benign (0) | pathogenic (1) | total |
|---|---|---|---|
| 1 | 70,177 | 8,637 | 78,814 |
| 2 | 53,869 | 10,185 | 64,054 |
| 3 | 3,611 | 2,346 | 5,957 |

### 4.6 By syndrome group (labelled SNVs, v2)
| syndrome_group | genes | labelled | pathogenic | % pathogenic |
|---|---|---|---|---|
| Breast_Ovarian_HR_repair | 16 | 30,017 | 5,318 | 17.7 |
| Neurocutaneous_rhabdoid_DICER1 | 11 | 19,592 | 3,384 | 17.3 |
| Lynch_MMR_polymerase | 11 | 18,872 | 2,486 | 13.2 |
| Fanconi_DNA_repair_chromosome_instability | 14 | 15,275 | 1,685 | 11.0 |
| Polyposis_hamartoma | 12 | 12,584 | 1,889 | 15.0 |
| Renal | 5 | 10,191 | 1,374 | 13.5 |
| Hematologic_predisposition | 13 | 8,868 | 557 | 6.3 |
| Melanoma_skin_telomere | 8 | 6,395 | 511 | 8.0 |
| Retinoblastoma_Wilms_overgrowth | 6 | 5,243 | 953 | 18.2 |
| Endocrine_MEN | 6 | 5,160 | 626 | 12.1 |
| Other_hereditary | 9 | 4,558 | 619 | 13.6 |
| Paraganglioma_pheo | 8 | 4,135 | 744 | 18.0 |
| Xeroderma_NER | 8 | 3,441 | 360 | 10.5 |
| Hereditary_gastric | 2 | 3,088 | 267 | 8.6 |
| Li_Fraumeni | 1 | 1,406 | 395 | 28.1 |

### 4.7 Missense depth per gene (labelled, v2) — matters for gene-held-out evaluation
- Genes with **< 30** missense: **60** of 130 · genes with **< 10 pathogenic** missense: **76** · genes with **0 pathogenic** missense: **20** · median missense per gene: 31 · top-5 genes hold **29.1%** of all missense.

(These replace the 58 / 72 / 13 figures quoted earlier, which were computed *before* the start-loss fix.)

Largest genes:
| gene | n | path |
|---|---|---|
| BRCA2 | 820 | 113 |
| BRCA1 | 659 | 212 |
| NF1 | 517 | 428 |
| MSH2 | 432 | 142 |
| NSD1 | 409 | 167 |
| TP53 | 387 | 234 |
| TSC2 | 364 | 179 |
| PTEN | 274 | 262 |

### 4.8 VUS file (what the model would actually be asked to score)
| variant_class | rows | % of file |
|---|---|---|
| missense | 160,023 | 91.0 |
| splice_region | 5,527 | 3.1 |
| utr5 | 2,581 | 1.5 |
| utr3 | 2,072 | 1.2 |
| intronic | 1,919 | 1.1 |
| synonymous | 1,423 | 0.8 |
| nonsense | 874 | 0.5 |
| splice_canonical | 676 | 0.4 |
| other_noncoding | 529 | 0.3 |
| start_lost | 172 | 0.1 |

**Important:** ~91% of VUS are missense — exactly the class where the v1 model is weakest (§8). Do not treat VUS scores as clinical evidence.

### 4.9 Conflicting file
| variant_class | rows | % of file |
|---|---|---|
| missense | 27,577 | 75.3 |
| synonymous | 3,832 | 10.5 |
| splice_region | 2,441 | 6.7 |
| intronic | 1,374 | 3.8 |
| splice_canonical | 433 | 1.2 |
| nonsense | 338 | 0.9 |
| utr5 | 331 | 0.9 |
| utr3 | 207 | 0.6 |
| start_lost | 55 | 0.2 |
| other_noncoding | 20 | 0.1 |

### 4.10 Indel file
Contains insertions, deletions and **multi-nucleotide substitutions** (`ref`/`alt` length ≠ 1 or both > 1), max 50 bp, labelled only.
| variant_class | benign (0) | pathogenic (1) | % pathogenic |
|---|---|---|---|
| synonymous | 10 | 0 | 0.0 |
| utr3 | 136 | 2 | 1.4 |
| intronic | 5,216 | 161 | 3.0 |
| utr5 | 184 | 25 | 12.0 |
| splice_region | 1,087 | 237 | 17.9 |
| missense | 71 | 65 | 47.8 |
| other_noncoding | 43 | 106 | 71.1 |
| coding_no_protein_annot | 231 | 2,895 | 92.6 |
| splice_canonical | 13 | 678 | 98.1 |
| nonsense | 3 | 898 | 99.7 |
| frameshift | 32 | 28,058 | 99.9 |
| start_lost | 0 | 2 | 100.0 |

(`frameshift` and `nonsense` are ≥ 99.7% pathogenic → trivial; its `missense` rows are MNVs causing a single amino-acid change.)

### 4.11 Funnel (v2 run #2, from `manifest_1.json`)
| Counter | Value |
|---|---|
| total rows | 9,223,698 |
| rows in panel genes | 879,854 |
| skipped: row names ≥2 panel genes (e.g. overlapping loci) | 67 |
| GRCh38 | 436,951 |
| germline | 419,070 |
| no VCF alleles | 1,672 |
| `ref == alt` skipped | 41 |
| non-SNV and out of indel scope (>50 bp) skipped | 1,370 |
| labelled benign SNV / pathogenic SNV | 127,657 / 21,168 |
| VUS SNV / conflicting SNV | 175,796 / 36,608 |
| labelled indels | 40,153 |
| dropped for stars < 1 (SNV + indel labelled) | 2,284 |
| labelled SNV off the primary transcript | 949 (in 103 genes) |

Accounting note: VUS/conflicting **indels** are intentionally discarded without a counter, so the funnel does not sum to the germline total.

### 4.12 Dataset integrity checks that passed (v2)
- CSV row counts equal manifest counts for all four files ✅
- All 130 panel genes present; every gene ≥ 30 labelled variants ✅
- 0 duplicate `(chrom,pos,ref,alt,gene)`; no variant appears in more than one of labelled/VUS/conflicting ✅
- `stars ≥ 1` for every labelled row ✅
- 0 `ref == alt`; 0 non-ACGT in labelled SNV ✅
- 0 conflicting `ref_aa` per codon on primary transcript ✅
- On the 103,711 variants shared with v1: **label agreement 100%**, gene agreement 100% ✅
- Transcript heterogeneity: 33 of 130 genes have >1 NM_ accession in labelled rows; only 949 labelled rows (0.64%) are off-primary, mostly non-protein classes. "Primary" = **most frequent in this data**, not necessarily MANE Select (for BRCA1 `NM_007294.4`, BRCA2 `NM_000059.4`, TP53 `NM_000546.6`, APC `NM_000038.6` it coincides with the standard transcript; check the others before citing).

### 4.13 Known remaining properties (not bugs)
- Class mix is dominated by easy rows (§4.4).
- Missense is thin for many genes (§4.7).
- 11 benign nonsense (incl. expert-panel BRCA2 p.Lys3326Ter, a known benign C-terminal truncation) and 97 pathogenic synonymous variants (likely splice-affecting): genuine biology/curation edge cases, kept.
- No 4-star (practice guideline) labelled rows in this panel.

---

## 5. The 130-gene panel

**Curated by me from the common hereditary-cancer-predisposition literature/panels; it is not an official panel.** Review it against your intended standard (ClinGen, NCCN, ACMG SF) before citing. Some symbols may differ between nomenclature versions (e.g. `ABRAXAS1`); the manifest lists genes with zero/low labelled counts (currently none).

| Group | # | Genes |
|---|---|---|
| Breast_Ovarian_HR_repair | 16 | BRCA1, BRCA2, PALB2, ATM, CHEK2, BARD1, BRIP1, RAD51C, RAD51D, RAD51B, NBN, MRE11, RAD50, XRCC2, RECQL, ABRAXAS1 |
| Li_Fraumeni | 1 | TP53 |
| Lynch_MMR_polymerase | 11 | MLH1, MSH2, MSH6, PMS2, EPCAM, MSH3, MLH3, PMS1, POLE, POLD1, MBD4 |
| Polyposis_hamartoma | 12 | APC, MUTYH, NTHL1, AXIN2, GREM1, BMPR1A, SMAD4, STK11, PTEN, RNF43, FAN1, RPS20 |
| Hereditary_gastric | 2 | CDH1, CTNNA1 |
| Melanoma_skin_telomere | 8 | CDKN2A, CDK4, BAP1, MITF, POT1, ACD, TERF2IP, TERT |
| Endocrine_MEN | 6 | MEN1, RET, CDKN1B, CDC73, AIP, PRKAR1A |
| Paraganglioma_pheo | 8 | SDHA, SDHB, SDHC, SDHD, SDHAF2, TMEM127, MAX, VHL |
| Renal | 5 | FH, FLCN, MET, TSC1, TSC2 |
| Neurocutaneous_rhabdoid_DICER1 | 11 | NF1, NF2, SMARCB1, LZTR1, SUFU, PTCH1, PTCH2, SMARCA4, SMARCE1, DICER1, SPRED1 |
| Retinoblastoma_Wilms_overgrowth | 6 | RB1, WT1, DIS3L2, CDKN1C, REST, NSD1 |
| Hematologic_predisposition | 13 | RUNX1, GATA2, CEBPA, DDX41, ETV6, ANKRD26, TINF2, DKC1, RTEL1, PARN, SAMD9, SAMD9L, SBDS |
| Fanconi_DNA_repair_chromosome_instability | 14 | FANCA, FANCB, FANCC, FANCD2, FANCE, FANCF, FANCG, FANCI, FANCL, FANCM, SLX4, BLM, WRN, RECQL4 |
| Xeroderma_NER | 8 | XPA, XPC, ERCC2, ERCC3, ERCC4, ERCC5, POLH, DDB2 |
| Other_hereditary | 9 | HOXB13, EXT1, EXT2, PHOX2B, BUB1B, PTPN11, CEP57, TRIP13, CBL |

**Removed from v1 (24 genes, not in the v2 panel):** AKT1, ALK, AR, BRAF, CCND1, CDKN2B, CHEK1, EGFR, ENG, ERBB2, ERBB3, ESR1, HRAS, KIT, KRAS, MDM2, MDM4, MTOR, NRAS, PDGFRA, PIK3CA, PIK3R1, ROS1, XRCC3. v1 labelled 15 genes as "Oncogene"/GI-stromal groups; those are somatic-driver genes, not hereditary-predisposition genes, so v1's "hereditary cancer" framing was inaccurate.

---

## 6. Feature engineering (v1 implementation) and v2 status

> **v2 CSVs contain no engineered features.** The block below documents what v1 computed (`bio_chemistry_engine.py` + builder), with verified defects, so the features can be rebuilt correctly for v2.

### 6.1 The 19 continuous model inputs (v1) — order matters
`ref_A, ref_C, ref_G, ref_T, alt_A, alt_C, alt_G, alt_T` (one-hot of **genomic-strand** nucleotides) · `rel_protein_pos` (= `aa_pos / hand-typed protein length`, **0.5 sentinel** if no protein change) · `is_transition` · `is_missense` · `is_nonsense` · `is_synonymous` · `delta_mw, delta_logp, delta_tpsa, delta_charge` (alt − ref amino-acid property, RDKit from side-chain-free amino-acid SMILES; charge hand-set at pH 7.4) · `tanimoto_chem_dist` (1 − Tanimoto of Morgan fingerprints, radius 2, 1024 bits) · `grantham_score`.
Only `delta_mw, delta_logp, delta_tpsa, delta_charge, grantham_score` are standardised (`StandardScaler` fit on the training split only ✅ no scaler leakage).
Plus two **embedded categorical** inputs: gene id (93 → 16-d) and syndrome id (23 → 8-d). Total network input = 19 + 16 + 8 = **43**.

### 6.2 Verified defects in v1 features
| # | Defect | Evidence |
|---|---|---|
| 1 | **Grantham table incomplete** — 59 of 190 amino-acid pairs present, **131 missing**, silently defaulted to 100 | `bio_chemistry_engine.GRANTHAM`; **41.1% of v1 missense rows used the default**; code returns **100** for Cys→Gly while the README claims 159 (true value 159) |
| 2 | Stop codon encoded as SMILES `O` (water); any change involving a stop gets Grantham = 200 (hard-coded) | source |
| 3 | `rel_protein_pos` uses hand-typed lengths; **152 rows > 1** (WT1 accounts for 79) | v1 CSV |
| 4 | **41,431 rows (34.8%)** have the sentinel 0.5 → indistinguishable from a real mid-protein position | v1 CSV |
| 5 | Nucleotide one-hots on the **genomic** strand, inconsistent for minus-strand genes (e.g. BRCA1) | builder uses VCF alleles |
| 6 | Intronic/splice/UTR variants have all-zero chemistry → the 41k "no protein annotation" rows are nearly featureless | §4.4 / v1 table |
| 7 | Gene and syndrome embeddings are redundant (syndrome is a function of gene) | `CANCER_GENES` dict |
| 8 | `bio_chemistry_engine.compute_mutation_features` is **dead legacy code** (uses `chrom` as gene key, references a missing `unified_128_gene_reference_panel.fna`, computes a "mutual information"/dn-ds score that is not used) | source |

### 6.3 How to rebuild Grantham correctly
Grantham (1974): `D = 50.723 · sqrt( 1.833·(c−c')² + 0.1018·(p−p')² + 0.000399·(v−v')² )` with (composition c, polarity p, volume v) per amino acid. **Check performed:** the formula with the values below reproduces Cys–Gly = 158.3 ≈ **159** (the published value). Values (c, p, v): S (1.42, 9.2, 32), R (0.65, 10.5, 124), L (0, 4.9, 111), P (0.39, 8.0, 32.5), T (0.71, 8.6, 61), A (0, 8.1, 31), V (0, 5.9, 84), G (0.74, 9.0, 3), I (0, 5.2, 111), F (0, 5.2, 132), Y (0.20, 6.2, 136), C (2.75, 5.5, 55), H (0.58, 10.4, 96), Q (0.89, 10.5, 85), N (1.33, 11.6, 56), K (0.33, 11.3, 119), D (1.38, 13.0, 54), E (0.92, 12.3, 83), M (0, 5.7, 105), W (0.13, 5.4, 170). *These values are from memory of the original paper; only Cys–Gly was checked — validate more pairs against the published matrix (or use BLOSUM62) before relying on them.*

### 6.4 v2 feature rebuild plan
(a) join true protein lengths from MANE/UniProt on `on_primary_transcript`; (b) full Grantham/BLOSUM62; (c) add population frequency (gnomAD), conservation (phyloP/GERP), published predictors (REVEL, AlphaMissense, CADD) or protein-language-model scores (ESM); (d) splice features for `splice_*`/`intronic` (SpliceAI, distance to exon boundary); (e) coding-strand nucleotides; (f) drop the 0.5 sentinel (use a missing-indicator); (g) drop `is_missense/is_nonsense/is_synonymous` as separate flags if `variant_class` is one-hot encoded.

---

## 7. Model architecture and training (v1, fully specified)

**Name:** `GenomeDLResNet` in `train_tabular_mlp.py`. Verified from code **and** from the shapes inside `models/pan_cancer_germline_resnet.pth` ✅ (they agree).

```
x_cont (19) ─┐
gene_id  ──► Embedding(93, 16) ─┤ concat → 43
syn_id   ──► Embedding(23, 8)  ─┘
   │
   ▼  input_proj : Linear(43→256) → LayerNorm(256) → GELU → Dropout(0.2)
   ▼  res1 : y = GELU( x + [Linear(256,256) → LN → GELU → Dropout(0.2) → Linear(256,256) → LN → Dropout(0.2)](x) )
   ▼  res2 : (same block, separate weights)
   ▼  head : Linear(256→64) → LayerNorm(64) → GELU → Dropout(0.1) → Linear(64→1)
   ▼  one logit  →  sigmoid  →  P(pathogenic)
```

| Item | Value (verified) |
|---|---|
| Trainable parameters | **295,305** (gene_embed 1,488 · syndrome_embed 184 · input_proj 11,520 · each residual block 131,584 · head 16,577) |
| Output | **one logit** (binary). **Not** a 5-class softmax (README claims otherwise) |
| Loss | `BCEWithLogitsLoss(pos_weight = n_neg / n_pos)` (weighted binary cross-entropy; **not** ordinal) |
| Optimiser | AdamW, lr 1e-3, weight decay 1e-4 |
| Schedule | CosineAnnealingLR, `T_max = epochs` |
| Epochs / batch | 25 / 256 |
| Seed | 42 (torch, numpy, sklearn splits) |
| Split | stratified **70 / 15 / 15** (README says 80/20): train 83,331 · val 17,857 · test 17,857 |
| Selection | best validation ROC-AUC; checkpoint reloaded for the test evaluation |
| Checkpoint contents | `model_state_dict`, `scaler`, `gene_to_id`, `best_val_auc` (= 0.9663). **Syndrome mapping and hyper-parameters are NOT saved** |
| Scaling | `StandardScaler` on the 5 biophysical columns, fit on train only |

### 7.1 Inference path (`predict.py`, v1) — verified from code
- Loads checkpoint; unknown gene → **silently** `gene_id = 0` (`gene_to_id.get(gene, 0)`).
- **`syn_id = 0` hard-coded** for every variant (the model was trained with each gene's own syndrome id) → train/inference mismatch.
- `rel_protein_pos` defaults to 0.5; the genomic position is only displayed, never used by the model.
- "5 ACMG classes" = thresholds on the sigmoid output: ≥0.90 → Class 4 · ≥0.65 → Class 3 · ≥0.35 → Class 2 · ≥0.10 → Class 1 · else Class 0. These are arbitrary cut-offs on an **uncalibrated** (class-weighted) probability.
- "Organ risk radar": `risk = min(99, max(0.15·base, base·prob))` where `base` is a **hard-coded** per-syndrome lifetime-risk table; unknown syndromes fall back to the breast/ovarian table. Not a learned or personalised risk.
- CLI arguments that exist: `--demo --gene --ref --alt --ref_aa --alt_aa --pos --weights`. The README's `--chrom` / `--patient` options **do not exist** (command fails). No VCF is ever read.

---

## 8. Evaluation findings (measured)

**Methods and caveats.** Torch was not installable in my sandbox, so (i) shipped-checkpoint numbers use a **numpy re-implementation of the forward pass** with the exact split (`train_test_split`, seed 42, stratified) — validated because it reproduces the README's AUC 0.9652 and sensitivity 92.35% exactly; (ii) leakage experiments use **scikit-learn HistGradientBoosting on the same 19 features (+ gene id)** as a stand-in for the ResNet, because the question is about features/splits, not the exact network. Treat (ii) as strong evidence about the *data design*, not as a measurement of the ResNet itself.

### 8.1 Reproduction of the shipped v1 model (threshold 0.5)
| Split | n | ROC-AUC | Sensitivity | Specificity | Precision |
|---|---|---|---|---|---|
| train | 83,331 | 0.9720 | 0.9316 | 0.8660 | 0.5541 |
| val | 17,857 | 0.9663 | 0.9158 | 0.8633 | 0.5448 |
| **test** | 17,857 | **0.9652** ✅ | **0.9235** ✅ | **0.8600** ❌ (README: 0.9480) | **0.5410** (omitted by README) |

Test confusion: FP = 2,121, FN = 207 (≈ 2,500 TP). Nearly half of "pathogenic" calls are false positives.

### 8.2 Per-class performance of the shipped model (test)
| Variant class | n | positives | AUC |
|---|---|---|---|
| missense | 1,384 | 602 | **0.8715** |
| no protein annotation | 6,231 | 803 | 0.8117 |
| nonsense | 1,290 | 1,281 | 0.9482 |
| synonymous | 8,948 | 17 | 0.7819 (only 17 positives — noisy) |

The 0.9652 overall AUC is inflated by mixing classes with very different base rates (synonymous 0.2% vs nonsense 99.2% pathogenic).

### 8.3 Do trivial baselines match it? (same test split)
| Model | ALL | missense | nonsense | synonymous | no-protein |
|---|---|---|---|---|---|
| A. Class prior only (4 numbers, no ML) | **0.9209** | 0.500 | 0.500 | 0.500 | 0.500 |
| B. Class prior + gene prior | 0.9443 | 0.733 | 0.879 | 0.755 | 0.678 |
| C. Logistic regression, 19 features, no gene | 0.9324 | 0.635 | 0.712 | 0.512 | 0.592 |
| D. Gradient boosting, 19 features, no gene | 0.9331 | 0.727 | 0.799 | 0.582 | 0.592 |
| E. Gradient boosting + gene id | 0.9691 | 0.934 | 1.000 | 0.832 | 0.811 |
| F. Shipped ResNet | 0.9652 | 0.872 | 0.948 | 0.782 | 0.812 |

- A lookup of four numbers already gets **0.921 (95% of the headline)**.
- Missense-only single-feature AUC (n = 1,384): Grantham 0.600 · Tanimoto distance 0.612 · ΔMW 0.624 · relative position 0.489.
- The ResNet is **not better than gradient boosting** on the same features (README's "+3.8% over MLPs" has no baseline in the repo ❓/❌).
- The big jump (missense 0.63–0.73 → 0.87–0.93) appears only when **gene id + position** are present → the model uses location, not "universal biophysics".

### 8.4 Leakage experiments (5-fold CV, gradient boosting; "block" = 25-aa windows held out, 5-kb windows for unannotated rows)
| Split | ALL | missense | nonsense | synonymous | no-protein |
|---|---|---|---|---|---|
| Random + gene id (what v1 does) | 0.9672 | **0.930** | 0.975 | 0.757 | 0.795 |
| Random, no gene id | 0.9370 | 0.774 | 0.856 | 0.654 | 0.592 |
| Position-block held out + gene id | 0.9622 | 0.902 | 0.949 | 0.704 | 0.768 |
| Position-block held out, no gene id | 0.9357 | 0.752 | 0.851 | 0.623 | 0.588 |
| **Gene held out** (unseen genes, no gene id) | 0.9313 | **0.708** | 0.692 | 0.596 | 0.562 |

- **30.6%** of test missense variants (424 / 1,384) share an exact codon with a *training* variant; the average label of *other* variants at that codon, with zero chemistry, gives **AUC 0.934** on those 424 → neighbour memorisation.
- Honest generalisation for missense on unseen genes is ≈ **0.71** AUC, not 0.87–0.93.
- **BRCA2 → BRCA1 transfer** (README: "87.38%, universal biophysics"): no script in the repo performs it; my reproduction (train BRCA2 only, test BRCA1) gives ≈ **0.82 overall** (driven by nonsense variants) and **0.585 on BRCA1 missense** ❌.

### 8.5 Calibration and the "ACMG class" mapping (shipped model, test)
- 10-bin Expected Calibration Error ≈ **0.10**; the class-weighted loss inflates probabilities.
- With the README's own thresholds: "Class 3 – Likely Pathogenic" is truly pathogenic **33%** of the time; "Class 4 – Pathogenic" **93%**. The README itself says ACMG "Likely pathogenic" means > 90% and "Pathogenic" > 99%, yet maps ≥ 0.90 to *Pathogenic* — internally inconsistent. ACMG/AMP classification is evidence-rule based, not a probability threshold.

---

## 9. Audit of the original v1 README claims

| # | v1 README claim | Verdict | Evidence |
|---|---|---|---|
| 1 | "80 hereditary cancer genes" | ❌ | data/checkpoint have 93; v2 has 130 |
| 2 | Test ROC-AUC 0.9652 / badge 96.52% | ✅ | §8.1 |
| 3 | ROC-AUC 0.9698 (97.0%) and recall 90.46% (section 6) | ❌ unreproduced | contradicts the 0.9652 elsewhere; no run found |
| 4 | Sensitivity 92.35% | ✅ | §8.1 |
| 5 | Specificity 94.80% | ❌ | measured 86.00%; precision 54.1% omitted |
| 6 | 80/20 split | ❌ | 70/15/15 in code |
| 7 | 359,731 raw entries partitioned into 5 files | ⚠️ | arithmetic holds (119,045+135,839+49,105+32,656+23,086), but only 4 CSVs ship; the 23,086 group has no file; indel/conflicting files have no producing script |
| 8 | 140,262 VUS (section 3) vs 135,839 elsewhere | ❌ internal contradiction | CSV and raw audit: 135,839 |
| 9 | 5-class softmax output, cross-entropy/ordinal loss | ❌ | single sigmoid logit, weighted BCE |
| 10 | Calibrated ACMG classes | ❌ | §8.5, §7.1 |
| 11 | "Learns universal laws of protein folding damage"; BRCA2→BRCA1 87.38% | ❌ | §8.3–8.4 |
| 12 | "Outperforms standard MLPs by +3.8% AUC" | ❌/❓ | no baseline in repo; gradient boosting ≥ ResNet |
| 13 | VCF ingestion, interval-tree gene funnel, coordinate→codon mapping | ❌ not implemented | `predict.py` never parses VCF; amino acids typed by hand |
| 14 | Scale to 20,000 genes by resizing the embedding | ❌ | embedding of an unseen gene is untrained noise; new genes need labelled data |
| 15 | Personalised organ-specific lifetime risk | ❌ | hard-coded table × probability |
| 16 | 23 syndromes, 93 genes extracted | ✅ | data & checkpoint |
| 17 | Cys61Gly demo locus chr17:43044295 C>G | ❌ | dataset has BRCA1 p.Cys61Gly at a different locus (43106487, A>C on the plus strand); 43044295 is the 3′ end of the gene |
| 18 | CLI quick-start with `--chrom`, `--patient` | ❌ | not defined arguments |
| 19 | Grantham Cys→Gly = 159 | ❌ in code | code returns 100 (table incomplete) |
| 20 | Training 89.6 s on RTX 3050; inference < 15 ms | ❓ | no GPU/torch in my environment |
| 21 | Loss curve 0.65 → 0.03 | ❓ | not re-trained (torch unavailable) |

---

## 10. What I could NOT verify
- I did **not** open the raw `variant_summary.txt.gz` myself (450 MB, outside my network allow-list). Raw-file facts come from the audit/extractor outputs **you ran**; I verified those for internal consistency and against v1.
- ClinVar **release identity/date** (only local file mtime recorded).
- Any **retraining** of the PyTorch model, GPU timing, inference latency, training-loss claims.
- The **panel's clinical validity** and the MANE status of "primary" transcripts beyond the four genes named.
- HGVS parsing against *every* real `Name` format (tested on representative cases and on the full data via integrity checks, not exhaustively).
- Whether v2 features/models will perform better — no v2 model exists yet.

---

## 11. Recommended next steps and acceptance criteria

**Pipeline**
1. Build v2 features (§6.4) on `on_primary_transcript == 1` rows (keep other rows with features marked missing).
2. Treat `start_lost`, `nonsense`, `splice_canonical`, `frameshift` as rule-like classes; **headline = missense and splice_region**.
3. Split by **gene** (and report a protein-region split as a second, easier test); pool out-of-fold predictions; report per-gene only for genes with enough variants (§4.7).
4. Use **AUPRC and AUC per class**, plus precision at a fixed sensitivity; calibrate probabilities (isotonic/Platt) on a held-out set.
5. Include baselines A–E of §8.3 in every report; a model must beat them on **gene-held-out missense**.
6. Re-run robustness with `--min-stars 2`.
7. Score VUS only after the above; never present as a diagnosis; drop "ACMG class" and "risk radar" wording or label it illustrative.

**Suggested "correct enough to publish" thresholds (my proposal, not a standard):** gene-held-out missense AUC well above the 0.71 gradient-boosting/no-gene baseline, with confidence intervals; ECE ≤ 0.05 after calibration; every number in the README reproducible by a script in the repo.

---

## 12. How to verify everything yourself

**12.1 Re-run the extraction (v2)**
```bash
python extract_clinvar_hereditary.py variant_summary.txt.gz --outdir out --min-stars 1 --indels
# compare out/manifest.json "counts" with §4.11
```
**12.2 Dataset integrity (pandas)**
```python
import pandas as pd, json
L = pd.read_csv("clinvar_labelled_snv.csv"); m = json.load(open("manifest.json"))
assert len(L) == m["counts"]["snv_benign"] + m["counts"]["snv_pathogenic"]
assert L.gene.nunique() == 130 and L.stars.min() >= 1
assert not L.duplicated(["chrom","pos","ref","alt","gene"]).any()
assert (L.ref != L.alt).all()
assert not ((L.variant_class == "missense") & (L.aa_pos == 1)).any()
P = L[L.on_primary_transcript == 1]
mm = P[P.variant_class.isin(["missense","nonsense","synonymous"])]
assert (mm.groupby(["gene","aa_pos"]).ref_aa.nunique() <= 1).all()
print(pd.crosstab(L.variant_class, L.label))
```
**12.3 v1 class-prior baseline (shows the headline is mostly easy rows)** — needs the v1 CSV (`pan_cancer_germline_dataset.csv`): derive class from `ref_aa=='.'`, `is_nonsense`, `is_synonymous`, `is_missense`, learn the 4 class prior rates on the stratified train split (seed 42), compute test AUC (expect ≈ 0.921).
**12.4 Re-evaluate the shipped checkpoint (needs torch; I did not run this snippet — I validated the same numbers with a numpy forward pass)**
```python
import torch, numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from train_tabular_mlp import prepare_pan_cancer_data, GenomeDLResNet
data = prepare_pan_cancer_data()
ck = torch.load("models/pan_cancer_germline_resnet.pth", weights_only=False)
model = GenomeDLResNet(19, data["num_genes"], data["num_syndromes"]); model.load_state_dict(ck["model_state_dict"]); model.eval()
Xc, g, s, y = data["test"]
with torch.no_grad():
    p = torch.sigmoid(model(torch.tensor(Xc, dtype=torch.float32), torch.tensor(g), torch.tensor(s))).squeeze().numpy()
tn, fp, fn, tp = confusion_matrix(y, p >= 0.5).ravel()
print("AUC", roc_auc_score(y, p), "sens", tp/(tp+fn), "spec", tn/(tn+fp), "prec", tp/(tp+fp))
# expect AUC 0.9652, sens 0.9235, spec 0.8600, prec 0.5410
```
**12.5 Parameter count:** `sum(p.numel() for p in model.parameters())` → 295,305.
**12.6 Check the CLI claim:** `python predict.py --chrom chr17` → argparse error (unrecognised argument).
**12.7 Grantham coverage:** count pairs of the 20 amino acids present in `bio_chemistry_engine.GRANTHAM` (expect 59 of 190); `get_grantham('C','G')` → 100.
**12.8 Leakage test sketch:** group-k-fold by `gene` vs stratified k-fold with a gradient-boosting classifier on the 19 features; compare missense AUC (expect ≈ 0.71 vs ≈ 0.93 with gene id).

---

## 13. Prompt template for another model

> "Below is a README describing a ClinVar-based hereditary-cancer variant classifier (v1 repo, v2 dataset). Do **not** trust it. For each ✅/❌ claim, tell me (a) the cheapest check to confirm it, (b) whether the stated number is plausible, and (c) any inconsistency you find between sections. Then critique: label circularity, class imbalance in difficulty, gene/position leakage, calibration, and whether the gene panel is defensible. List what extra evidence you would need."

---

## 14. Glossary
**ClinVar** NCBI archive of variant–disease classifications · **SNV** single-nucleotide variant · **VUS** variant of uncertain significance · **P/LP, B/LB** (likely) pathogenic / benign · **ACMG/AMP** evidence-based variant-classification guidelines (not probability thresholds) · **HGVS** variant naming standard (`c.` coding DNA, `p.` protein) · **MANE Select** curated standard transcript per gene · **ROC-AUC / AUPRC** ranking metrics (AUPRC is more informative under class imbalance) · **ECE** expected calibration error · **Grantham score** amino-acid substitution distance from composition, polarity, volume · **Gene-held-out split** test genes never seen in training · **Leakage** information from the test set (e.g. same codon) present in training · **Primary transcript** (here) the most frequent transcript per gene in this dataset.
