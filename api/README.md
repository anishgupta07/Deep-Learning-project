# NCBI & ENA Genomic Data Harvesting API Module
=================================================

This repository module contains clean, standalone API clients and utilities for querying, searching, and downloading:
1. **NCBI Entrez Gene Reference Panels** (via NCBI Entrez E-utilities with 10 req/s API Key support).
2. **Patient SRA Sequencing Runs** (via European Nucleotide Archive / ENA Portal API).

---

## 1. Quick Start

### Installation
Make sure you have Python 3.9+ and the required packages installed:
```bash
pip install requests biopython
```

### Run Instant API Verification Test
Verify connectivity and your NCBI API Key in under 5 seconds:
```bash
cd api
python quick_test_api.py
```

---

## 2. Configuration (`config.json`)

All API settings, search queries, and credentials are centralized in [`config.json`](file:///m:/cancer-test/api/config.json):

```json
{
  "ncbi": {
    "email": "monojoycodes@gmail.com",
    "api_key": "2e92c37a9ad293b99ca3b8a94fef3073fb09",
    "rate_limit_req_per_sec": 10,
    "default_gene_query": "(oral[All Fields] AND carcinoma[All Fields]) AND \"Homo sapiens\"[Organism] AND alive[prop]",
    "batch_size": 100
  },
  "ena_sra": {
    "api_endpoint": "https://www.ebi.ac.uk/ena/portal/api/search",
    "default_study_query": "tax_eq(9606) AND study_title=\"oral squamous cell carcinoma\"",
    "fields": "run_accession,fastq_ftp,fastq_bytes,read_count",
    "format": "json"
  }
}
```

> [!TIP]
> Your NCBI API Key is already configured in `config.json` to allow up to **10 requests per second** (compared to 3 req/sec without a key).

---

## 3. How to Use the API Clients

### A. Downloading Reference Genes ([`download_genes_api.py`](file:///m:/cancer-test/api/download_genes_api.py))

Fetches sliced genomic FASTA sequences for top disease-associated genes directly from NCBI RefSeq:

#### Example 1: Download default 50 Oral Carcinoma genes
```bash
python download_genes_api.py 50 my_oral_genes.fna
```

#### Example 2: Download 100 genes for any custom cancer type (e.g., Lung Cancer)
```bash
python download_genes_api.py "(lung[Title/Abstract] AND carcinoma[Title/Abstract]) AND \"Homo sapiens\"[Organism]" lung_cancer_100_genes.fna
```

#### Example 3: Use as a Python Module in Your Own Scripts
```python
from download_genes_api import fetch_reference_genes

# Download 20 Oral Carcinoma genes
ref_file = fetch_reference_genes(
    query='(oral[All Fields] AND carcinoma[All Fields]) AND "Homo sapiens"[Organism]',
    target_count=20,
    output_fasta='oral_20_genes.fna'
)
print("Saved reference panel to:", ref_file)
```

---

### B. Searching & Streaming Patient SRA Datasets ([`download_sra_api.py`](file:///m:/cancer-test/api/download_sra_api.py))

Queries the European Nucleotide Archive (ENA / EBI SRA Mirror) to discover patient runs and direct `.fastq.gz` download streams:

#### Example 1: Discover top 10 patient SRA runs
```bash
python download_sra_api.py 10
```

#### Example 2: Stream and download the first sample
```bash
python download_sra_api.py 5 --download
```

#### Example 3: Use as a Python Module in Your Own Scripts
```python
from download_sra_api import search_ena_runs, stream_sra_sample

# Search for 5 patient runs
runs = search_ena_runs(query='tax_eq(9606) AND study_title="oral squamous cell carcinoma"', limit=5)

for r in runs:
    print(f"Sample Accession: {r['accession']}, URL: {r['fastq_url']}, Total Reads: {r['read_count']:,}")

# Stream 50,000 real reads for the first sample into FASTA format
first_run = runs[0]
stream_sra_sample(first_run['fastq_url'], f"patient_{first_run['accession']}.fasta", max_reads=50000)
```

---

## 4. API Technical Details & Workflow

```
                          [ NCBI & ENA API WORKFLOW ]

  1. NCBI Gene Search:
     esearch.fcgi?db=gene&term=<query>&api_key=... ──────► Returns Gene IDs List
                                                                     │
  2. RefSeq Coordinate Extractor:                                    ▼
     esummary.fcgi?db=gene&id=<batch_ids> ───────────────► Extracts (ChrAccVer, ChrStart, ChrStop)
                                                                     │
  3. Sliced Sequence Fetch:                                          ▼
     efetch.fcgi?db=nuccore&id=...&seq_start=...&seq_stop=... ──► Consolidated Reference FASTA (.fna)

  4. ENA SRA Query:
     https://www.ebi.ac.uk/ena/portal/api/search ────────► Returns Direct .fastq.gz Stream URLs
```

---

## 5. File Inventory

| File | Description |
| :--- | :--- |
| [`config.json`](file:///m:/cancer-test/api/config.json) | Centralized API credentials, endpoints, queries, and rate limits. |
| [`download_genes_api.py`](file:///m:/cancer-test/api/download_genes_api.py) | NCBI Entrez Gene reference harvester with sliced FASTA download. |
| [`download_sra_api.py`](file:///m:/cancer-test/api/download_sra_api.py) | ENA / SRA patient sequencing search and FASTQ streaming client. |
| [`quick_test_api.py`](file:///m:/cancer-test/api/quick_test_api.py) | Automated 5-second API connection and verification test suite. |
| [`README.md`](file:///m:/cancer-test/api/README.md) | Complete documentation and integration guide. |
