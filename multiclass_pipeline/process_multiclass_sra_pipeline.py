#!/usr/bin/env python3
"""
Multi-Class SRA Patient Harvester & Disk-Safe VCF Pipeline (Optimal 25 Samples/Class)
====================================================================================
Harvests balanced patient cohorts across 11 cancer classes from ENA/SRA,
streams real FASTQ sequence reads, applies Strategy 1 targeted filtering,
calls variants, and outputs class-tagged VCF files.
"""

import sys
import os
import json
import time
import requests
import gzip
import gc
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory for variant calling engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generate_vcf import process_reads_and_call_variants, build_kmer_index

MAX_WORKERS = 4
GLOBAL_KMER_INDEX = None
REF_SEQ_STR = None
TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), "cancer_taxonomy.json")

def load_taxonomy():
    with open(TAXONOMY_PATH, "r") as f:
        return json.load(f)["cancers"]

def load_and_index_reference_panel(ref_fasta, k=17):
    global GLOBAL_KMER_INDEX, REF_SEQ_STR
    abs_ref_fasta = os.path.abspath(ref_fasta)
    print(f"Loading and indexing multi-class reference panel ({abs_ref_fasta})...")
    seq_chunks = []
    with open(abs_ref_fasta, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.startswith(">"):
                seq_chunks.append(line.strip().upper())
    REF_SEQ_STR = "".join(seq_chunks)
    print(f"Loaded multi-class reference sequence ({len(REF_SEQ_STR):,} bp)")
    GLOBAL_KMER_INDEX = build_kmer_index(REF_SEQ_STR, k=k, stride=3)
    print(f"Indexed {len(GLOBAL_KMER_INDEX):,} unique target {k}-mers for Strategy 1 filtering.")

def fetch_multiclass_ena_runs(samples_per_cancer=25):
    cancers = load_taxonomy()
    endpoint = "https://www.ebi.ac.uk/ena/portal/api/search"
    
    all_runs = []
    print("=" * 75)
    print(f"DISCOVERING BALANCED MULTI-CLASS PATIENT SRA RUNS ({samples_per_cancer} Samples / Class)")
    print("=" * 75)
    
    for c in cancers:
        cid = c["class_id"]
        cname = c["name"]
        query = c["sra_query"]
        params = {
            "result": "read_run",
            "query": query,
            "fields": "run_accession,fastq_ftp,fastq_bytes,read_count",
            "limit": samples_per_cancer,
            "format": "json"
        }
        
        try:
            r = requests.get(endpoint, params=params, timeout=30)
            data = r.json()
            found = 0
            for item in data:
                acc = item.get("run_accession", "")
                ftp_str = item.get("fastq_ftp", "")
                if acc and ftp_str:
                    ftp_url = "https://" + ftp_str.split(";")[0]
                    all_runs.append({
                        "class_id": cid,
                        "class_name": cname,
                        "accession": acc,
                        "fastq_url": ftp_url
                    })
                    found += 1
            print(f"[Class {cid:02d}: {cname:<25}] Found {found} SRA Patient Runs.")
        except Exception as e:
            print(f"[Class {cid:02d}: {cname:<25}] Error: {e}")
            
    print("\n" + "=" * 75)
    print(f"Total Discovered Multi-Class SRA Cohort: {len(all_runs)} Patient Runs")
    print("=" * 75)
    return all_runs

def download_and_filter_sra_reads(fastq_url, output_fasta, k=17, max_reads=100000):
    """Strategy 1: Streams FASTQ over HTTP and retains ONLY matching target reads"""
    retained_reads = 0
    try:
        r = requests.get(fastq_url, stream=True, timeout=60)
        with open(output_fasta, "w", encoding="utf-8") as out_f:
            with gzip.open(r.raw, "rt", encoding="utf-8", errors="ignore") as gz:
                header = ""
                for idx, line in enumerate(gz):
                    if idx % 4 == 0:
                        header = line.strip().replace("@", ">")
                    elif idx % 4 == 1:
                        r_seq = line.strip().upper()
                        if len(r_seq) >= k:
                            matched = False
                            for offset in range(0, min(len(r_seq) - k + 1, 60), 15):
                                if r_seq[offset:offset + k] in GLOBAL_KMER_INDEX:
                                    matched = True
                                    break
                            if matched:
                                out_f.write(f"{header}\n{r_seq}\n")
                                retained_reads += 1
                                if retained_reads >= max_reads:
                                    break
        return retained_reads
    except Exception as e:
        print(f"   [Error] Stream failed: {e}")
        return 0

def process_single_multiclass_sample(sample_item, sample_idx, total, ref_fasta, vcf_dir):
    cid = sample_item["class_id"]
    cname_clean = sample_item["class_name"].replace(" ", "_")
    acc = sample_item["accession"]
    fastq_url = sample_item["fastq_url"]
    
    abs_ref_fasta = os.path.abspath(ref_fasta)
    os.makedirs(vcf_dir, exist_ok=True)
    
    vcf_filename = f"class_{cid:02d}_{cname_clean}_sample_{sample_idx:04d}_{acc}.vcf"
    vcf_path = os.path.join(vcf_dir, vcf_filename)
    temp_fasta = f"temp_multi_{acc}.fasta"
    
    # Checkpoint check: skip if already completed and valid (> 2 KB)
    if os.path.exists(vcf_path) and os.path.getsize(vcf_path) > 2000:
        return {"acc": acc, "class_id": cid, "status": "SKIPPED", "vcf": vcf_path}
        
    print(f"[{sample_idx}/{total}] [Class {cid:02d}: {sample_item['class_name']}] Processing {acc}...")
    
    # 1. Strategy 1 Targeted Read Stream
    retained_cnt = download_and_filter_sra_reads(fastq_url, temp_fasta)
    file_size_mb = os.path.getsize(temp_fasta) / (1024 * 1024) if os.path.exists(temp_fasta) else 0
    print(f"   [Strategy 1] Extracted {retained_cnt:,} target reads -> {temp_fasta} ({file_size_mb:.2f} MB)")
    
    # 2. Call Variants
    try:
        process_reads_and_call_variants(abs_ref_fasta, temp_fasta, vcf_path, min_dp=2, min_vaf=0.15)
        vcf_size_kb = os.path.getsize(vcf_path) / 1024 if os.path.exists(vcf_path) else 0
        print(f"   [SUCCESS] VCF written: {vcf_filename} ({vcf_size_kb:.1f} KB)")
        status = "SUCCESS"
    except Exception as err:
        print(f"   [Error] Variant calling failed: {err}")
        status = f"FAILED ({err})"
        
    # 3. DISK CLEANUP - Delete temp raw reads immediately
    if os.path.exists(temp_fasta):
        os.remove(temp_fasta)
        
    gc.collect()
    return {"acc": acc, "class_id": cid, "status": status, "vcf": vcf_path}

def run_multiclass_pipeline(ref_fasta="multiclass_cancer_reference_panel.fna", samples_per_cancer=25, vcf_dir="vcf_output_multiclass"):
    os.makedirs(vcf_dir, exist_ok=True)
    load_and_index_reference_panel(ref_fasta)
    
    runs = fetch_multiclass_ena_runs(samples_per_cancer=samples_per_cancer)
    total = len(runs)
    
    print("=" * 75)
    print(f"EXECUTING MULTI-CLASS SRA PIPELINE ({total} Total Patient Runs)")
    print(f"Reference Panel:  {ref_fasta}")
    print(f"Output Directory: {vcf_dir}/")
    print(f"Parallel Workers: {MAX_WORKERS} Threads (Intel i5-13500H)")
    print(f"Disk Protection:  ACTIVE (< 3.4 GB Total Storage)")
    print("=" * 75)
    
    start_time = time.time()
    completed = 0
    skipped = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_multiclass_sample, item, idx + 1, total, ref_fasta, vcf_dir): item
            for idx, item in enumerate(runs)
        }
        
        for future in as_completed(futures):
            res = future.result()
            status = res["status"]
            if status == "SKIPPED":
                skipped += 1
                completed += 1
            elif status == "SUCCESS":
                completed += 1
                
            new_processed = completed - skipped
            if new_processed > 0:
                elapsed = time.time() - start_time
                avg_sec = elapsed / new_processed
                rem_hrs = (avg_sec * (total - completed)) / 3600.0
                print(f"   [Multi-Class Progress] {completed}/{total} done (Resumed: {skipped}) | ETA: {rem_hrs:.2f} hours")

    print("\n" + "=" * 75)
    print("SUCCESS: Multi-Class SRA Pipeline Completed!")
    print(f"Total VCFs Generated: {completed:,}/{total:,}")
    print(f"Output VCF Directory: {vcf_dir}/")
    print("=" * 75)

if __name__ == "__main__":
    samples_per_type = 25
    ref_panel = "multiclass_cancer_reference_panel.fna"
    
    if len(sys.argv) > 1:
        samples_per_type = int(sys.argv[1])
    if len(sys.argv) > 2:
        ref_panel = sys.argv[2]
        
    run_multiclass_pipeline(ref_fasta=ref_panel, samples_per_cancer=samples_per_type)
