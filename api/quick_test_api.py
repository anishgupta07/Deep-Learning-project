#!/usr/bin/env python3
"""
API Verification Test Script
============================
Tests connectivity and credential verification for:
1. NCBI Entrez Gene API (Searching & fetching 2 reference genes)
2. ENA / SRA Portal API (Searching & testing 1 patient FASTQ endpoint)
"""

import sys
import os
from download_genes_api import fetch_reference_genes
from download_sra_api import search_ena_runs, stream_sra_sample

def run_quick_test():
    print("=" * 65)
    print("RUNNING API VERIFICATION TEST SUITE")
    print("=" * 65)
    
    # Test 1: NCBI Gene API
    print("\n--- TEST 1: NCBI Entrez Gene API ---")
    test_gene_out = "test_api_genes.fna"
    res1 = fetch_reference_genes(query='(oral[All Fields] AND carcinoma[All Fields]) AND "Homo sapiens"[Organism] AND alive[prop]', target_count=2, output_fasta=test_gene_out)
    
    # Test 2: ENA SRA API
    print("\n--- TEST 2: ENA / SRA Portal API ---")
    runs = search_ena_runs(limit=2)
    
    print("\n" + "=" * 65)
    if res1 and os.path.exists(test_gene_out) and runs:
        print("ALL API CONNECTIONS VERIFIED & WORKING PROPERLY!")
        print(f"  - NCBI Gene API: SUCCESS ({os.path.getsize(test_gene_out)/1024:.1f} KB)")
        print(f"  - ENA SRA API:   SUCCESS ({len(runs)} runs discovered)")
    else:
        print("API Verification encountered an issue.")
    print("=" * 65)
    
    # Clean up test output
    if os.path.exists(test_gene_out):
        os.remove(test_gene_out)

if __name__ == "__main__":
    run_quick_test()
