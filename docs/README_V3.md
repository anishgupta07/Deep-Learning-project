# GenomeDL — Hereditary-Cancer Variant Pathogenicity (v2 dataset, verified README)

> One README for the whole repo. It replaces the old `README.md` (v1) and `README_V2.md`.
> Every number below was re-computed from the files in this repo unless it is explicitly marked
> **(carried over, not re-run)**.

---

## 0. Verdict (read this first)

| Question | Answer |
|---|---|
| Is the **v2 labelled CSV** (`clinvar_labelled_snv.csv`, 148,825 rows) correct? | **Yes, with 2 small known defects** (§4): 79 rows (0.05 %) carry the annotation of a *neighbouring non-panel gene*, and 20 stop-loss rows are named `missense`. Everything else passes. |
| Are the **mutations** themselves correct? | **Yes.** All 148,506 rows with a coding HGVS name agree with their VCF alleles on a single, biologically correct strand per gene; all 98,819 exonic protein-changing rows are reproduced exactly by the stated single-base change; genomic order matches transcript order for every row (§3). |
| Is the **featured CSV** (`v2_clinvar_featured_snv.csv`) correct? | **Mostly.** The 19 base columns are byte-identical to the labelled CSV and the Grantham/RDKit features recompute exactly, but 4 design problems remain (§5), mainly the approximate protein length and plus-strand nucleotide one-hots. |
| Do the **model scripts** work on v2? | **No, not as shipped.** `train_tabular_mlp.py` / `predict.py` are still v1 code and will fail on the v2 CSV (§6). **The model has not been retrained on v2.** |
| Is this a clinical tool? | **No.** Research prototype. Labels are ClinVar curator calls, so results mean "agreement with ClinVar", not ground-truth biology. |

**Limits of this verification.** The ClinVar source file, a GRCh38 reference genome, PyTorch and the v1 checkpoint were not
available, so I could **not** (a) re-run the extractor on the raw file, (b) compare each REF base to the genome, or
(c) re-evaluate the model. What *can* be proven from the CSV alone is proven in §3; the reference-allele check is replaced by
an indirect one (codon reconstruction, §3.4).

---

## 1. Project goal

- **Task:** given a single-nucleotide variant (SNV) in one of 130 hereditary-cancer-predisposition genes, predict whether ClinVar calls it
  **pathogenic / likely pathogenic (label 1)** or **benign / likely benign (label 0)**.
- **Not achieved / out of scope:** clinical diagnosis, ACMG/AMP classification, personalised lifetime risk, VCF ingestion, unseen genes.
- **Circularity:** ClinVar curators use rules such as "null variant ⇒ pathogenic" and "synonymous ⇒ benign", so a model partly re-learns those rules.

---

## 2. Repository map

| File | Role | Status |
|---|---|---|
| `extract_clinvar_hereditary.py` | Streams ClinVar `variant_summary.txt.gz` → 4 CSVs + `manifest.json` (stdlib only) | v2, works (2 defects, §4) |
| `build_v2_features.py` | Adds 20 feature columns → `v2_clinvar_featured_snv.csv` | v2, works (4 issues, §5) |
| `clinvar_labelled_snv.csv` | 148,825 labelled SNVs (0/1) | **verified** |
| `v2_clinvar_featured_snv.csv` | labelled CSV + features (39 columns) | verified, see §5 |
| `manifest.json` | parameters, counts, per-gene table, primary transcripts | matches the CSV |
| `train_tabular_mlp.py` | `GenomeDLResNet` + training loop | **v1 code**, incompatible with v2 CSV |
| `predict.py` | CLI inference + "risk report" | **v1 code**, several defects (§6) |
| `bio_chemistry_engine.py` | RDKit amino-acid properties (v1) | legacy; incomplete Grantham table |
| `requirements.txt`, `LICENSE` (MIT) | dependencies, license | fine |

Not part of this upload: `clinvar_vus_snv.csv`, `clinvar_conflicting_snv.csv`, `clinvar_labelled_indels.csv`, the v1 CSVs and
the v1 checkpoint. Statements about those files are carried over from the previous audit and are marked as such.

---

## 3. Dataset verification — what was checked and what passed

Run `verify_dataset.py` (Appendix A) to reproduce every row of this table.

### 3.1 Basic integrity — all PASS
| Check | Result |
|---|---|
| Row count equals manifest (`snv_benign + snv_pathogenic`) | 148,825 = 127,657 + 21,168 ✅ |
| 130 genes, 15 syndrome groups, per-group counts | all match ✅ |
| Duplicate `(chrom,pos,ref,alt,gene)` / duplicate `variation_id` | 0 / 0 ✅ |
| `ref != alt`, only A/C/G/T | ✅ |
| Each gene on exactly one chromosome (1–22, X) | ✅ |
| `label` agrees with `clinical_significance` text | 0 mismatches ✅ (12 distinct strings; 6 rows carry extras such as "risk factor") |
| `stars ≥ 1`, and `review_status → stars` mapping (single submitter 1, multiple 2, expert panel 3) | ✅ |
| Only missing values: `transcript` | 320 rows (319 are genomic-only `NC_…g.` names) |

### 3.2 Mutation correctness — HGVS vs VCF alleles — all PASS
- 148,506 rows have a coding HGVS name `c.…X>Y`. In **every** one, `X>Y` equals the VCF `ref>alt` (plus-strand gene) or its complement (minus-strand gene). **0 rows fit neither.**
- Per gene the strand is unique: **65 plus-strand and 65 minus-strand genes**, 0 mixed. This agrees with known biology (BRCA1, TP53, PALB2, CHEK2, PMS2, MUTYH are minus-strand; BRCA2, ATM, APC, PTEN, MLH1 are plus-strand).
- The 319 rows with only a genomic name (`NC_…:g.pos ref>alt`): `g.pos`, REF and ALT equal the CSV in 319/319.

### 3.3 Mutation correctness — protein change — all PASS
Tested on the 98,819 exonic coding rows whose HGVS gene is the panel gene (missense 9,746 · nonsense 10,454 · start-lost 196 · synonymous 78,423):
- `aa_pos == ceil(c.position / 3)` → **98,819 / 98,819**.
- Using the standard genetic code and the position inside the codon (`(c.pos-1) mod 3`), **a codon for `ref_aa` exists in which the stated coding-strand base change gives exactly `alt_aa`** (or a stop / the same amino acid) → **98,819 / 98,819, 0 failures**.

### 3.4 Indirect reference-allele check — PASS
Without a genome file, REF bases were checked through codons: for 66,722 codons (gene, transcript, aa_pos), 11,892 have variants at ≥ 2 codon positions. In all of them the observed reference bases combine into a codon that really codes for `ref_aa`, and no codon position has two different reference bases (0 conflicts). A wrong REF or a shifted coordinate would break this.

### 3.5 Coordinates (GRCh38) — PASS
- For 151 (gene, transcript) groups covering 148,333 rows, sorting by c. position gives monotone genomic position in the gene's strand direction → **0 rows out of order**.
- Spot checks of well-known variants from memory of the literature (not from a reference file): BRCA1 c.181T>G p.Cys61Gly at `17:43106487 A>C`, TP53 p.Arg175His `17:7675088 C>T`, p.Arg248Gln `17:7674220 C>T`, p.Arg273His `17:7673802 C>T`, p.Arg248Trp `17:7674221 G>A` are all present, labelled 1 and `missense`.
- The data ranges of BRCA1, BRCA2, PTEN, ATM and APC fall inside their GRCh38 gene coordinates (±20 kb).

### 3.6 Class composition (matches README_V2 §4.4 cell by cell)
| variant_class | benign | pathogenic | total | % path |
|---|---|---|---|---|
| synonymous | 78,344 | 97 | 78,441 | 0.1 |
| intronic | 32,976 | 296 | 33,272 | 0.9 |
| nonsense | 11 | 10,444 | 10,455 | 99.9 |
| missense | 5,751 | 4,008 | 9,759 | 41.1 (41.0 after §4 corrections) |
| splice_region | 7,451 | 458 | 7,909 | 5.8 |
| splice_canonical | 3 | 5,603 | 5,606 | 99.9 |
| utr3 / utr5 | 1,459 / 1,344 | 6 / 56 | 1,465 / 1,400 | 0.4 / 4.0 |
| other_noncoding | 315 | 4 | 319 | 1.3 |
| start_lost | 0 | 196 | 196 | 100 |
| coding_no_protein_annot | 3 | 0 | 3 | 0.0 |

**How to read it:** 75 % of rows are synonymous + intronic (almost all benign) and nonsense / canonical-splice / start-lost are ≥ 99.9 % pathogenic, so a headline metric over all rows is dominated by easy rows. The hard classes are **missense** and **splice_region**. Other counts (stars, syndrome groups, missense depth per gene: 60 genes with < 30 missense, 20 with no pathogenic missense) also reproduce.

---

## 4. Defects found in the labelled CSV (small, but real)

### D1 — 79 rows carry another gene's annotation (0.05 %)
ClinVar lists overlapping loci as `GeneSymbol = "MSH3;DHFR"`. `panel_gene()` in the extractor keeps the one panel gene, but `Name` (and therefore `variant_class`, `ref_aa`, `alt_aa`, `aa_pos`, `transcript`) describes the **other** gene.

| panel gene ← HGVS gene (rows) |
|---|
| MSH3 ← DHFR (17) · PRKAR1A ← FAM20A (8) · PTEN ← KLLN (8) · FANCI ← POLG (6) · MAX ← FNTB (5) · MUTYH ← TOE1 (5) · RTEL1 ← TNFRSF6B (4) · MSH6 ← FBXO11 (4) · NF1 ← EVI2B (3), EVI2A (2), OMG (1) · FANCA ← ZNF276 (2) · 12 genes with 1 row each (CTNNA1, FAN1, CDK4, CDC73, ACD, ERCC2, FANCB, MSH2, FANCL, POLE, SMARCB1, TSC2) |

- 32 of the 79 carry protein fields (13 missense, 18 synonymous, 1 nonsense); labels 72 benign / 7 pathogenic; all have `on_primary_transcript = 0`.
- Effect: e.g. a POLG missense counted as a FANCI missense. Small, but wrong.
- **Fix** (in the extractor, after `classify`): extract the gene in `Name` with `re.search(r"\(([^)]+)\):c\.", name)` and **drop the row (or blank `variant_class`/aa fields) when it differs from the panel gene.**

### D2 — 20 stop-loss variants are named `missense`
`PROT_RE` accepts `p.Ter214Leu`, and `classify()` falls through to `missense` with `ref_aa = "*"` (e.g. VHL p.Ter214Leu, PTEN p.Ter404Cys, MEN1 p.Ter611Arg). All 20 are pathogenic.
- After removing D1 and D2, **true missense = 9,726 rows, 3,986 pathogenic (41.0 %)** instead of 9,759 / 4,008 / 41.1 %.
- **Fix:** add `if ref3 == "Ter": return "stop_lost", ...` before the missense return.

### Minor notes (not errors)
- 6 genomic-only (`NC_`) `other_noncoding` rows lie > 50 kb outside the gene's annotated span (TP53 `17:7579817` ≈ 88 kb away, FAN1 `15:30801393`, 4 TERF2IP rows). They are probably distal entries; they carry no protein features. Review them or drop them.
- 949 labelled rows are off the "primary" transcript (79 of them are D1). "Primary" means *most frequent in this data*, not MANE Select.
- No 4-star (practice-guideline) rows exist in this panel; 11 benign nonsense and 97 pathogenic synonymous rows are genuine curation edge cases and are kept.
- The 130-gene panel is curated by hand, not an official panel; check it against ClinGen / NCCN / ACMG before citing.

---

## 5. The featured CSV (`v2_clinvar_featured_snv.csv`, built by `build_v2_features.py`)

**What it is:** the labelled CSV plus 20 columns: `has_protein_change, rel_protein_pos, grantham_score, delta_mw, delta_charge, delta_logp, delta_tpsa, tanimoto_chem_dist, is_transition, gene_id, syndrome_id, variant_class_id, ref_A, alt_A, ref_C, alt_C, ref_G, alt_G, ref_T, alt_T` (39 columns, same 148,825 rows).

### 5.1 Verified correct
| Check | Result |
|---|---|
| First 19 columns identical to `clinvar_labelled_snv.csv` | 0 differences ✅ |
| `grantham_score` recomputed with the Grantham (1974) formula | max abs difference **0.0** over all standard-amino-acid missense rows ✅ |
| Formula vs the old hand-typed v1 table (59 pairs) | 57 / 59 within ±1.5; the two outliers (Phe–Trp, Gln–Asn) look like errors in the *old* table (published values ≈ 40 and ≈ 46, from memory; formula gives 39.7 and 45.8) ✅ |
| Cys→Gly Grantham | 158.2 (published 159; v1 code returned 100) ✅ |
| `is_transition`, one-hot `ref_*` / `alt_*` equal `ref` / `alt` | ✅ |
| `gene_id`, `syndrome_id`, `variant_class_id` follow the sorted / declared order, 1-to-1 | ✅ |
| No NaN except `transcript` (320) | ✅ |
| Stop-gain rows get Grantham 200, Tanimoto 1.0 | as designed |

### 5.2 Issues
| # | Issue | Evidence | Fix |
|---|---|---|---|
| F1 | **"Canonical protein length" is the maximum observed `aa_pos`**, not the real length. | BRCA1 1864 (real 1863), BRCA2 3419 (3418), PTEN 404 (403), ATM 3057 (3056), APC 2844 (2843); 11 of 16 genes I checked are +1, 5 exact (UniProt lengths from memory). For sparse genes it can be much too short. 186 rows hit the cap `rel_protein_pos = 1.0`. | Join MANE / UniProt lengths. |
| F2 | **start_lost is scored like a missense** (p.Met1Val → Grantham 21). | All 196 start_lost rows have Grantham 10.1–94.5 although they are 100 % pathogenic. | Keep `variant_class_id` as a feature; set a fixed severity for start_lost. |
| F3 | **The 32 protein-bearing D1 rows** get protein features of the wrong gene (`has_protein_change = 1`). | §4 D1 | Fix D1 in the extractor and re-run. |
| F4 | **Nucleotide one-hots are plus-strand**, inconsistent for the 65 minus-strand genes. | §3.2 | Complement `ref`/`alt` for minus-strand genes (list below) before one-hot encoding. |

**Minus-strand genes (65, derived from the data):** ABRAXAS1, ACD, ANKRD26, AXIN2, BAP1, BARD1, BRCA1, BRIP1, CDK4, CDKN1C, CDKN2A, CEBPA, CHEK2, DDX41, DICER1, ERCC2, ERCC3, EXT1, FANCA, FANCB, FANCC, FANCF, FANCG, FANCL, FH, FLCN, GATA2, HOXB13, MAX, MBD4, MEN1, MLH3, MRE11, MUTYH, NBN, NTHL1, PALB2, PARN, PHOX2B, PMS2, POLE, POT1, PTCH1, PTCH2, RAD51D, RECQL, RECQL4, RNF43, RPS20, RUNX1, SAMD9, SAMD9L, SBDS, SDHB, SLX4, SMARCE1, TERT, TINF2, TMEM127, TP53, TSC1, WT1, XPA, XPC, XRCC2. The other 65 panel genes are plus-strand.

**Also note:** synonymous rows get `has_protein_change = 1` with all-zero chemistry (fine, but not "protein-changing"); intronic / UTR / splice rows have all-zero chemistry by design — use `variant_class_id` to tell them apart.

---

## 6. Code status (v1 code that still ships)

### 6.1 `train_tabular_mlp.py`
- Reads `data/pan_cancer_germline_dataset.csv` (v1) and needs columns `cancer_syndrome`, `is_missense`, `is_nonsense`, `is_synonymous`. The v2 featured CSV has `syndrome_group` and **no** `is_*` flags → `KeyError`. It also needs the v1 file to exist.
- To train on v2: read `v2_clinvar_featured_snv.csv`, use `syndrome_id`, replace the three `is_*` flags with one-hot `variant_class_id`, and add `rel_protein_pos`'s missing-indicator (`has_protein_change`).
- **Split:** random stratified 70/15/15 on variants. This leaks neighbours (same codon, same gene). Use a gene-held-out split (e.g. `GroupKFold` by `gene`) for the headline number, and report per-class AUC/AUPRC.

### 6.2 `predict.py`
- `syn_id = 0` hard-coded for every variant (training used each gene's own syndrome id); unknown genes silently become `gene_id = 0`.
- "5 ACMG classes" are arbitrary thresholds on an **uncalibrated** class-weighted probability; ACMG/AMP is evidence-rule based. The "organ risk radar" is a hard-coded table multiplied by the probability, not a learned risk.
- The demo variant `BRCA1 p.Cys61Gly` uses locus `chr17:43044295 C>G`. That locus is **not** in the dataset; the real variant is `17:43106487 A>C` (plus strand; c.181T>G on the minus-strand gene).
- The old README's `--chrom` and `--patient` options **do not exist** in the argparse (only `--demo --gene --ref --alt --ref_aa --alt_aa --pos --weights`). No VCF is ever read.
- It imports `bio_chemistry_engine.py`, whose Grantham table has only 59 of 190 pairs and defaults the rest to 100, so inference would not match the v2 features.

### 6.3 `bio_chemistry_engine.py`
Legacy. `compute_mutation_features` is dead code (references a missing `unified_128_gene_reference_panel.fna`, uses `chrom` as the gene key, computes an unused dN/dS and "mutual information" score). Stop codon is encoded as SMILES `O` (water). Replace its Grantham lookup with the matrix in `build_v2_features.py`.

---

## 7. Model architecture (v1 checkpoint, unchanged, **not retrained on v2**)

```
x_cont (19) ─┐
gene_id  ──► Embedding(93, 16) ─┤ concat → 43
syn_id   ──► Embedding(23, 8)  ─┘
   ▼ input_proj: Linear(43→256) → LayerNorm → GELU → Dropout(0.2)
   ▼ res1, res2: y = GELU( x + [Linear → LN → GELU → Dropout → Linear → LN → Dropout](x) )
   ▼ head: Linear(256→64) → LayerNorm → GELU → Dropout(0.1) → Linear(64→1)
   ▼ one logit → sigmoid → P(pathogenic)
```
295,305 parameters · weighted BCE (`pos_weight = n_neg/n_pos`) · AdamW lr 1e-3, wd 1e-4 · cosine schedule · 25 epochs, batch 256 · seed 42 · split 70/15/15 · `StandardScaler` on 5 biophysical columns, fit on train only. Output is **one** logit, not a 5-class softmax. A v2 model needs `num_genes = 130`, `num_syndromes = 15`.

**Measured v1 results (carried over, not re-run — no PyTorch / v1 data here):** test ROC-AUC 0.9652, sensitivity 92.35 %, specificity 86.0 % (old README said 94.8 %), precision 54.1 %. Per class: missense AUC 0.87, no-protein-annotation 0.81, nonsense 0.95, synonymous 0.78. A four-number class-prior baseline already reaches AUC 0.921; gene-held-out missense AUC is ≈ 0.71 versus 0.87–0.93 with random splits, so the headline is inflated by easy rows and neighbour leakage.

---

## 8. Old v1 README — claims that are false or unsupported
80 genes (data has 93; v2 has 130) · test split 80/20 (code: 70/15/15) · specificity 94.8 % · ROC-AUC 0.9698 / recall 90.46 % (contradicts 0.9652) · 140,262 VUS (CSV: 135,839) · 5-class softmax / ordinal loss (code: one sigmoid logit) · "calibrated" ACMG classes · BRCA2→BRCA1 "87.38 %, universal biophysics" (no script; a reproduction gave ≈ 0.585 on BRCA1 missense) · "+3.8 % over MLPs" (no baseline; gradient boosting is as good) · VCF ingestion / interval trees / scaling to 20,000 genes (not implemented) · personalised organ risk (hard-coded table) · `--chrom/--patient` CLI · Grantham Cys→Gly = 159 (code returned 100) · demo locus `43044295 C>G` (wrong). Items such as GPU timing (89.6 s, < 15 ms) and the loss curve were never re-checked.

---

## 9. v1 → v2 changes (short)
| Aspect | v1 | v2 |
|---|---|---|
| Genes | 93 (incl. somatic drivers KRAS, EGFR, BRAF…) | 130 hereditary-predisposition genes, 15 groups |
| Review filter | none | `--min-stars 1` (benign/pathogenic only) |
| Labelled SNVs | 119,045 (100,997 / 18,048) | **148,825** (127,657 / 21,168) |
| VUS / conflicting / indels | 135,839 / 32,656 / 49,105 (no script) | 175,796 / 36,608 / 40,153 (script) — **carried over** |
| Non-protein variants | one lump, features all zero | explicit `variant_class` (11 classes) |
| Start-loss | counted as missense | `start_lost` (196 rows) |
| Transcript | mixed | `transcript`, `on_primary_transcript` |
| `ref == alt` rows | present | removed (41) |
| Provenance | none | `manifest.json` |
| Features | in CSV (v1 defects) | separate `build_v2_features.py` (Grantham full, §5) |
| Model | ResNet | **unchanged, not retrained** |

Source: ClinVar `variant_summary.txt.gz` (450,727,547 bytes, mtime 2026-10-05; the real release date is not recorded — note it yourself). Labels: pathogenic = text contains "pathogenic" and none of conflict / uncertain / benign; benign = the mirror; GRCh38, germline, `len(ref) == len(alt) == 1`.

---

## 10. Recommended next steps
1. **Fix the extractor:** drop or blank rows whose HGVS gene ≠ panel gene (D1); add `stop_lost` (D2); re-run extractor and `build_v2_features.py`.
2. **Features:** real protein lengths (MANE/UniProt), coding-strand nucleotides, start-lost severity, missing-indicator instead of a 0.5 sentinel; consider gnomAD frequency, conservation (phyloP/GERP), REVEL / AlphaMissense / ESM scores, SpliceAI for splice/intronic.
3. **Training:** adapt `train_tabular_mlp.py` to the v2 columns (§6.1); gene-held-out split; report AUC **and** AUPRC per class (headline = missense and splice_region); keep the baselines (class prior, class + gene prior, gradient boosting with and without `gene_id`) in every report; calibrate with isotonic / Platt.
4. **Robustness:** re-run with `--min-stars 2`.
5. **Scoring VUS** only after the above, never as a diagnosis; drop the "ACMG class" and "risk radar" wording or label them illustrative.

---

## 11. Quick start
```bash
pip install -r requirements.txt
python extract_clinvar_hereditary.py variant_summary.txt.gz --outdir data --min-stars 1 --indels
python build_v2_features.py --input data/clinvar_labelled_snv.csv --output data/v2_clinvar_featured_snv.csv
python verify_dataset.py data/clinvar_labelled_snv.csv data/manifest.json     # Appendix A
# training / inference scripts are v1 code: adapt them first (§6)
```

---

## Appendix A — `verify_dataset.py` (reproduces §3)
Save as `verify_dataset.py` and run `python verify_dataset.py clinvar_labelled_snv.csv manifest.json`. Expected output: all `PASS`, plus `INFO … 79` (D1) and `INFO … 20` (D2).

```python
import sys, re, json, itertools, pandas as pd, numpy as np
csv_path, man_path = sys.argv[1], sys.argv[2]
L = pd.read_csv(csv_path, dtype={"chrom": str}); man = json.load(open(man_path))
ok = lambda name, cond, extra="": print(("PASS  " if cond else "FAIL  ") + name + (f"  [{extra}]" if extra else ""))
c = man["counts"]
ok("row count == manifest", len(L) == c["snv_benign"] + c["snv_pathogenic"], len(L))
ok("label counts == manifest", (L.label == 0).sum() == c["snv_benign"] and (L.label == 1).sum() == c["snv_pathogenic"])
ok("130 genes, 15 groups", L.gene.nunique() == 130 and L.syndrome_group.nunique() == 15)
ok("no duplicate variant / variation_id", not L.duplicated(["chrom", "pos", "ref", "alt", "gene"]).any() and not L.variation_id.duplicated().any())
ok("ref != alt, only ACGT", ((L.ref != L.alt) & L.ref.isin(list("ACGT")) & L.alt.isin(list("ACGT"))).all())
ok("every gene on a single chromosome", (L.groupby("gene").chrom.nunique() == 1).all())
s = L.clinical_significance.str.lower()
exp = np.where(s.str.contains("pathogenic") & ~s.str.contains("benign|conflict|uncertain"), 1,
      np.where(s.str.contains("benign") & ~s.str.contains("pathogenic|conflict|uncertain"), 0, -9))
ok("label agrees with clinical_significance text", (exp == L.label).all())
rx = re.compile(r":c\.(?P<csg>[-*]?)(?P<cpos>\d+)(?P<coff>[+-]\d+)?(?P<cr>[ACGT])>(?P<ca>[ACGT])")
L = pd.concat([L, L.hgvs_name.str.extract(rx)], axis=1)
L["name_gene"] = L.hgvs_name.str.extract(r"\(([^)]+)\):c\.")[0]
H = L[L.cr.notna()]
print(f"INFO  rows whose HGVS belongs to a different (non-panel) gene: {(H.name_gene != H.gene).sum()}")
G = H[H.name_gene == H.gene].copy()
comp = dict(zip("ACGT", "TGCA"))
G["plus"] = (G.cr == G.ref) & (G.ca == G.alt); G["minus"] = (G.cr == G.ref.map(comp)) & (G.ca == G.alt.map(comp))
ok("c. alleles == VCF alleles (plus) or complement (minus)", (G.plus | G.minus).all())
st = G.groupby("gene")[["plus", "minus"]].sum()
ok("each gene has exactly one strand", ((st.plus == 0) | (st.minus == 0)).all(), f"{(st.minus == 0).sum()} plus / {(st.plus == 0).sum()} minus")
aas = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
code = {a + b + d: aas[i] for i, (a, b, d) in enumerate(itertools.product("TCAG", repeat=3))}
P = G[(G.csg == "") & G.coff.isna() & G.variant_class.isin(["missense", "synonymous", "nonsense", "start_lost"])].copy()
P["cpos"] = P.cpos.astype(int); P["cp"] = (P.cpos - 1) % 3
ok("aa_pos == ceil(c.pos / 3)", ((P.cpos - 1) // 3 + 1 == P.aa_pos).all(), len(P))
def reachable(ra, aa, cp, x, y):
    for cod in (k for k, v in code.items() if v == ra and k[cp] == x):
        if code[cod[:cp] + y + cod[cp + 1:]] == aa: return True
    return False
P["feas"] = [reachable(r, a, cp, x, y) for r, a, cp, x, y in zip(P.ref_aa, P.alt_aa, P.cp, P.cr, P.ca)]
ok("ref_aa -> alt_aa produced by the stated single-base change", P.feas.all(), f"{(~P.feas).sum()} failures")
sl = L[(L.variant_class == "missense") & (L.ref_aa == "*")]
print(f"INFO  rows classed 'missense' that are really stop-loss (p.Ter->aa): {len(sl)}")
```

---

## Glossary
**ClinVar** NCBI archive of variant classifications · **SNV** single-nucleotide variant · **VUS** variant of uncertain significance · **P/LP, B/LB** (likely) pathogenic / benign · **HGVS** variant naming (`c.` coding DNA, `p.` protein, `g.` genomic) · **MANE Select** curated standard transcript per gene · **Grantham score** amino-acid distance from composition, polarity and volume · **Gene-held-out split** test genes never seen in training · **Leakage** test information (same codon / gene) present in training · **AUPRC** area under the precision-recall curve (better than ROC-AUC under imbalance) · **ECE** expected calibration error.

*License: MIT. Maintainer: Anish Gupta.*
