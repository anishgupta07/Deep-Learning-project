# GenomeDL v3 Data Assets & Provenance

All dataset paths are anchored under `data/` using GRCh38 coordinates and canonical transcript annotations across the 130 hereditary cancer predisposition genes.

## Directory Layout

### `reference/` — Static Biological Reference Assets
- **`panel.json`**: Single source of truth for all 130 panel genes (`syndrome_group`, `strand`, `protein_length`, `mane_transcript`). Derived from UniProt / MANE Select canonical transcripts and ClinVar coding-strand concordance.
- **`panel_coordinates.json`**: GRCh38 genomic coordinate intervals (`chrom`, `start`, `end`) for each of the 130 panel genes, derived from Ensembl GRCh38 gene boundaries.
- **`gene_strand.json`**: Coding strand orientation (`"+"` or `"-"`) per gene (`130` genes).
- **`protein_lengths.json`**: Canonical CDS amino acid sequence lengths per gene (`130` genes, MANE Select / UniProt Swiss-Prot reviewed).
- **`panel_proteins.fasta`**: Canonical human amino acid sequences (`130` proteins) fetched via UniProtKB REST API (`organism_id:9606 AND reviewed:true`).
- **`penetrance.csv`**: Curated published lifetime carrier cancer risk ranges and non-cancer syndromic features with literature sources (e.g., Kuchenbaecker et al. JAMA 2017, NCCN / Chompret et al., Olivier et al.) and evidence levels.

### `processed/` — Labelled & Featurized ClinVar Datasets
- **`clinvar_labelled_v3.csv`**: `148,747` ClinVar SNVs filtered to GRCh38, germline origin, `>= 1` review star, and the 130-gene panel (foreign-gene HGVS rows removed and stop-loss variants typed as `stop_lost`).
- **`v3_featured_snv.csv`**: `148,747` SNVs with `49` columns combining biophysical RDKit/Grantham descriptors, coding-strand one-hots, categorical indices, and ESM-2 masked-marginal features (`esm_llr`, `esm_available`).

### `cache/` — Precomputed ESM-2 Protein Language Model Cache
- **`esm_cache.parquet`**: Precomputed `facebook/esm2_t12_35M_UR50D` residue-level amino acid log-probabilities and 480-D masked hidden state vectors, tracked in Git.
- **`esm_embeddings.pt`** *(gitignored local tensor)*: `(148747, 480)` float16 tensor reconstructed automatically on fresh clones from `esm_cache.parquet` and `v3_featured_snv.csv` via `genomedl.features.esm_features.load_or_reconstruct_embeddings`.

### `demo/` — Synthetic Verification Input
- **`sample_demo.vcf`**: Synthetic GRCh38 VCF (`5` panel variants covering known pathogenic controls, a common polymorphism, a heterozygous recessive *MUTYH* variant, and an unresolved *PALB2* variant).
