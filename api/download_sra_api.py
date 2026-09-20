#!/usr/bin/env python3
"""
ENA / SRA Patient Sequencing Downloader API Client
==================================================
Queries the European Nucleotide Archive (ENA / EBI SRA Mirror) API
to search patient cohorts and retrieve direct FASTQ sequencing streams.
"""

import sys
import os
import json
import requests
import gzip

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {
        "ena_sra": {
            "api_endpoint": "https://www.ebi.ac.uk/ena/portal/api/search",
            "default_study_query": 'tax_eq(9606) AND study_title="oral squamous cell carcinoma"',
            "fields": "run_accession,fastq_ftp,fastq_bytes,read_count",
            "format": "json"
        }
    }

def search_ena_runs(query=None, limit=10):
    config = load_config()["ena_sra"]
    endpoint = config.get("api_endpoint", "https://www.ebi.ac.uk/ena/portal/api/search")
    
    if not query:
        query = config.get("default_study_query")
        
    params = {
        "result": "read_run",
        "query": query,
        "fields": config.get("fields", "run_accession,fastq_ftp,fastq_bytes,read_count"),
        "limit": limit,
        "format": "json"
    }
    
    print("=" * 65)
    print("ENA / SRA COHORT SEARCH API CLIENT")
    print(f"Query Endpoint: {endpoint}")
    print(f"Search Query:   {query}")
    print(f"Limit:          {limit} Samples")
    print("=" * 65)
    
    try:
        r = requests.get(endpoint, params=params, timeout=30)
        data = r.json()
        
        runs = []
        for item in data:
            acc = item.get("run_accession", "")
            ftp_str = item.get("fastq_ftp", "")
            read_count = item.get("read_count", "0")
            if acc and ftp_str:
                # Primary FASTQ URL
                ftp_url = "https://" + ftp_str.split(";")[0]
                runs.append({
                    "accession": acc,
                    "fastq_url": ftp_url,
                    "read_count": int(read_count) if str(read_count).isdigit() else 0
                })
                
        print(f"\nFound {len(runs)} Available SRA Patient Sequencing Runs:")
        for idx, item in enumerate(runs[:10], 1):
            print(f"  {idx:02d}. {item['accession']} -> {item['fastq_url']} ({item['read_count']:,} reads)")
            
        return runs
    except Exception as e:
        print(f"Error querying ENA Portal API: {e}")
        return []

def stream_sra_sample(fastq_url, output_fasta, max_reads=50000):
    """Streams gzipped FASTQ from ENA and converts into FASTA file"""
    print(f"\nStreaming {max_reads:,} reads from {fastq_url} -> {output_fasta}...")
    try:
        r = requests.get(fastq_url, stream=True, timeout=60)
        count = 0
        with open(output_fasta, "w", encoding="utf-8") as out_f:
            with gzip.open(r.raw, "rt", encoding="utf-8", errors="ignore") as gz:
                header = ""
                for idx, line in enumerate(gz):
                    if idx % 4 == 0:
                        header = line.strip().replace("@", ">")
                    elif idx % 4 == 1:
                        seq = line.strip().upper()
                        out_f.write(f"{header}\n{seq}\n")
                        count += 1
                        if count >= max_reads:
                            break
                            
        file_size_mb = os.path.getsize(output_fasta) / (1024 * 1024)
        print(f"SUCCESS: Streamed {count:,} reads -> {output_fasta} ({file_size_mb:.2f} MB)")
        return output_fasta
    except Exception as e:
        print(f"Error streaming FASTQ: {e}")
        return None

if __name__ == "__main__":
    limit = 5
    if len(sys.argv) > 1:
        limit = int(sys.argv[1])
        
    runs = search_ena_runs(limit=limit)
    if runs and len(sys.argv) > 2 and sys.argv[2] == "--download":
        sample = runs[0]
        stream_sra_sample(sample["fastq_url"], f"sample_{sample['accession']}.fasta", max_reads=10000)
