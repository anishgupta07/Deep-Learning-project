#!/usr/bin/env python3
"""
NCBI Entrez Gene Reference Downloader API Client
================================================
Fetches targeted human cancer reference genes from NCBI Gene and Nuccore 
databases using BioPython Entrez with API Key support (10 req/s).
"""

import sys
import os
import json
import time
from Bio import Entrez

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.json")

def load_config():
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, "r") as f:
            return json.load(f)
    return {
        "ncbi": {
            "email": "monojoycodes@gmail.com",
            "api_key": "2e92c37a9ad293b99ca3b8a94fef3073fb09",
            "default_gene_query": '(oral[All Fields] AND carcinoma[All Fields]) AND "Homo sapiens"[Organism] AND alive[prop]',
            "batch_size": 100
        }
    }

def fetch_reference_genes(query=None, target_count=50, output_fasta="downloaded_reference_genes.fna"):
    config = load_config()["ncbi"]
    Entrez.email = config.get("email", "user@example.com")
    Entrez.api_key = config.get("api_key", "")
    
    if not query:
        query = config.get("default_gene_query")
        
    batch_size = config.get("batch_size", 100)
    
    print("=" * 65)
    print("NCBI GENE HARVESTING API CLIENT")
    print(f"Query:        {query}")
    print(f"Target Count: {target_count:,} Genes")
    print(f"Output File:  {output_fasta}")
    print(f"API Key:      {'ACTIVE (10 req/s)' if Entrez.api_key else 'INACTIVE (3 req/s)'}")
    print("=" * 65)
    
    # 1. Search NCBI Gene
    print("\n1. Searching NCBI Gene Database...")
    search_handle = Entrez.esearch(db="gene", term=query, retmax=target_count, sort="relevance")
    search_res = Entrez.read(search_handle)
    search_handle.close()
    
    gene_ids = search_res["IdList"]
    print(f"   Found {len(gene_ids):,} Gene IDs matching query.")
    
    if not gene_ids:
        print("   No genes found. Exiting.")
        return None
        
    # 2. Extract Genomic Coordinates
    print("\n2. Extracting RefSeq Genomic Coordinates...")
    gene_slices = []
    for i in range(0, len(gene_ids), batch_size):
        batch = gene_ids[i:i + batch_size]
        summary_handle = Entrez.esummary(db="gene", id=",".join(batch))
        summaries = Entrez.read(summary_handle)
        summary_handle.close()
        
        for doc in summaries["DocumentSummarySet"]["DocumentSummary"]:
            gene_id = doc.attributes.get("uid", "")
            symbol = doc.get("Name", "Unknown")
            loc_hist = doc.get("GenomicInfo", [])
            
            if loc_hist and len(loc_hist) > 0:
                primary_loc = loc_hist[0]
                acc = primary_loc.get("ChrAccVer")
                start = primary_loc.get("ChrStart")
                stop = primary_loc.get("ChrStop")
                
                if acc and start is not None and stop is not None:
                    p1, p2 = sorted([int(start) + 1, int(stop) + 1])
                    gene_slices.append((symbol, gene_id, acc, p1, p2))
                    
        time.sleep(0.1 if Entrez.api_key else 0.35)
        
    print(f"   Located exact chromosome coordinates for {len(gene_slices):,}/{len(gene_ids):,} genes.")
    
    # 3. Sliced efetch FASTA sequence download
    print(f"\n3. Downloading Genomic FASTA Sequences to {output_fasta}...")
    total_bp = 0
    saved_count = 0
    
    with open(output_fasta, "w", encoding="utf-8") as out_f:
        for idx, (symbol, gid, acc, start, stop) in enumerate(gene_slices, 1):
            success = False
            for attempt in range(3):
                try:
                    handle = Entrez.efetch(
                        db="nuccore",
                        id=acc,
                        seq_start=start,
                        seq_stop=stop,
                        rettype="fasta",
                        retmode="text"
                    )
                    lines = handle.read().strip().splitlines()
                    handle.close()
                    
                    if lines:
                        # Clean standardized FASTA header
                        lines[0] = f">{symbol}_GeneID{gid}_{acc}:{start}-{stop}"
                        out_f.write("\n".join(lines) + "\n\n")
                        seq_len = sum(len(l.strip()) for l in lines[1:])
                        total_bp += seq_len
                        saved_count += 1
                        success = True
                        break
                except Exception as e:
                    time.sleep(1.0)
                    
            if idx % 25 == 0 or idx == len(gene_slices):
                print(f"   Downloaded {idx:,}/{len(gene_slices):,} genes ({total_bp:,} bp)...")
                
            time.sleep(0.1 if Entrez.api_key else 0.35)
            
    file_size_mb = os.path.getsize(output_fasta) / (1024 * 1024)
    print("\n" + "=" * 65)
    print("SUCCESS: Gene Reference Panel Download Complete!")
    print(f"Genes Written:      {saved_count:,}")
    print(f"Total DNA Bases:    {total_bp:,} bp")
    print(f"Output File:        {output_fasta} ({file_size_mb:.2f} MB)")
    print("=" * 65)
    return output_fasta

if __name__ == "__main__":
    count = 50
    out_file = "downloaded_reference_genes.fna"
    query = None
    
    if len(sys.argv) > 1:
        try:
            count = int(sys.argv[1])
        except ValueError:
            query = sys.argv[1]
    if len(sys.argv) > 2:
        out_file = sys.argv[2]
        
    fetch_reference_genes(query=query, target_count=count, output_fasta=out_file)
