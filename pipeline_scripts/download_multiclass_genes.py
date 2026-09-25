#!/usr/bin/env python3
"""
Multi-Class Cancer Reference Panel Downloader
=============================================
Harvests unique reference driver genes across all 11 cancer classes from NCBI 
and compiles a unified multi-cancer reference panel FASTA.
"""

import sys
import os
import json
import time
from Bio import Entrez

NCBI_EMAIL = "monojoycodes@gmail.com"
NCBI_API_KEY = "2e92c37a9ad293b99ca3b8a94fef3073fb09"
TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), "cancer_taxonomy.json")

def load_taxonomy():
    with open(TAXONOMY_PATH, "r") as f:
        return json.load(f)["cancers"]

def download_multiclass_reference_panel(genes_per_cancer=50, output_fasta="multiclass_cancer_reference_panel.fna"):
    Entrez.email = NCBI_EMAIL
    Entrez.api_key = NCBI_API_KEY
    cancers = load_taxonomy()
    
    print("=" * 70)
    print("MULTI-CLASS CANCER REFERENCE PANEL HARVESTER (11 CANCER TYPES)")
    print(f"Target Genes per Class: {genes_per_cancer}")
    print(f"Output Reference File:  {output_fasta}")
    print(f"NCBI API Key:           {'ACTIVE (10 req/s)' if Entrez.api_key else 'INACTIVE'}")
    print("=" * 70)
    
    seen_gene_ids = set()
    all_gene_slices = [] # (symbol, gid, acc, start, stop, cancer_name)
    
    # 1. Search and extract coordinates for each cancer class
    for c in cancers:
        cid = c["class_id"]
        cname = c["name"]
        query = c["gene_query"]
        print(f"\n[Class {cid:02d}: {cname}] Searching NCBI Gene Database...")
        
        try:
            handle = Entrez.esearch(db="gene", term=query, retmax=genes_per_cancer, sort="relevance")
            res = Entrez.read(handle)
            handle.close()
            gene_ids = res.get("IdList", [])
            print(f"   Found {len(gene_ids)} gene candidates.")
            
            # Fetch summaries
            if gene_ids:
                s_handle = Entrez.esummary(db="gene", id=",".join(gene_ids))
                summaries = Entrez.read(s_handle)
                s_handle.close()
                
                added_for_cancer = 0
                for doc in summaries["DocumentSummarySet"]["DocumentSummary"]:
                    gid = doc.attributes.get("uid", "")
                    symbol = doc.get("Name", "Unknown")
                    loc_hist = doc.get("GenomicInfo", [])
                    
                    if gid not in seen_gene_ids and loc_hist and len(loc_hist) > 0:
                        primary_loc = loc_hist[0]
                        acc = primary_loc.get("ChrAccVer")
                        start = primary_loc.get("ChrStart")
                        stop = primary_loc.get("ChrStop")
                        
                        if acc and start is not None and stop is not None:
                            p1, p2 = sorted([int(start) + 1, int(stop) + 1])
                            all_gene_slices.append((symbol, gid, acc, p1, p2, cname))
                            seen_gene_ids.add(gid)
                            added_for_cancer += 1
                            
                print(f"   Added {added_for_cancer} new unique reference genes for {cname}.")
            time.sleep(0.1)
        except Exception as e:
            print(f"   Error searching {cname}: {e}")
            
    print("\n" + "=" * 70)
    print(f"Total Unique Driver Genes to Download: {len(all_gene_slices)} across 11 Cancers")
    print("=" * 70)
    
    # 2. Download Sliced FASTA sequences
    total_bp = 0
    saved_genes = 0
    with open(output_fasta, "w", encoding="utf-8") as out_f:
        for idx, (symbol, gid, acc, start, stop, cname) in enumerate(all_gene_slices, 1):
            success = False
            for attempt in range(3):
                try:
                    h = Entrez.efetch(db="nuccore", id=acc, seq_start=start, seq_stop=stop, rettype="fasta", retmode="text")
                    lines = h.read().strip().splitlines()
                    h.close()
                    if lines:
                        # Header with cancer tag
                        lines[0] = f">{symbol}_GeneID{gid}_{acc}:{start}-{stop} [Cancer: {cname}]"
                        out_f.write("\n".join(lines) + "\n\n")
                        seq_len = sum(len(l.strip()) for l in lines[1:])
                        total_bp += seq_len
                        saved_genes += 1
                        success = True
                        break
                except Exception:
                    time.sleep(0.5)
            if idx % 25 == 0 or idx == len(all_gene_slices):
                print(f"   Downloaded {idx:,}/{len(all_gene_slices):,} genes ({total_bp:,} bp)...")
            time.sleep(0.1)
            
    file_size_mb = os.path.getsize(output_fasta) / (1024 * 1024)
    print("\n" + "=" * 70)
    print("SUCCESS: Multi-Class Reference Panel Created!")
    print(f"Total Unique Genes: {saved_genes:,}")
    print(f"Total DNA Length:   {total_bp:,} bp")
    print(f"Output File:        {output_fasta} ({file_size_mb:.2f} MB)")
    print("=" * 70)
    return output_fasta

if __name__ == "__main__":
    count_per_cancer = 50
    if len(sys.argv) > 1:
        count_per_cancer = int(sys.argv[1])
    download_multiclass_reference_panel(genes_per_cancer=count_per_cancer)
