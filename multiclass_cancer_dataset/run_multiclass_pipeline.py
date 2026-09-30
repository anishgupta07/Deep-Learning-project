#!/usr/bin/env python3
"""
Automated Multi-Class Pipeline Orchestrator
===========================================
Streams patient SRA runs across all 11 cancer classes, saves VCFs into their 
respective class subdirectories, and compiles the master Deep Learning dataset.
"""

import sys
import os
import glob
import json
import time
import requests
import gzip
import gc
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory for variant caller & dataset builder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generate_vcf import process_reads_and_call_variants, build_multi_kmer_index, load_multi_reference
from build_dl_dataset import build_dl_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REF_PANEL = os.path.join(BASE_DIR, "reference_panel", "multiclass_cancer_reference_panel.fna")
PATIENT_VCF_DIR = os.path.join(BASE_DIR, "patient_vcfs")
MASTER_CSV = os.path.join(BASE_DIR, "master_dataset", "multiclass_cancer_master_dataset.csv")
MASTER_SCHEMA = os.path.join(BASE_DIR, "master_dataset", "multiclass_dataset_schema.json")
TAXONOMY_PATH = os.path.join(BASE_DIR, "pipeline_scripts", "cancer_taxonomy.json")

MAX_WORKERS = 4
GLOBAL_KMER_INDEX = None
GLOBAL_CONTIGS = None

def load_taxonomy():
    with open(TAXONOMY_PATH, "r") as f:
        return json.load(f)["cancers"]

def index_reference():
    global GLOBAL_KMER_INDEX, GLOBAL_CONTIGS
    print(f"Loading and indexing reference panel: {REF_PANEL}...")
    GLOBAL_CONTIGS = load_multi_reference(REF_PANEL)
    GLOBAL_KMER_INDEX = build_multi_kmer_index(GLOBAL_CONTIGS, k=17, stride=3)
    print(f"Loaded {len(GLOBAL_CONTIGS)} genes ({sum(len(c[0]) for c in GLOBAL_CONTIGS.values()):,} bp), indexed {len(GLOBAL_KMER_INDEX):,} 17-mers.")

def fetch_sra_runs_for_cancer(cancer_item, limit=10):
    endpoint = "https://www.ebi.ac.uk/ena/portal/api/search"
    query = cancer_item["sra_query"]
    params = {
        "result": "read_run",
        "query": query,
        "fields": "run_accession,fastq_ftp,fastq_bytes,read_count",
        "limit": limit,
        "format": "json"
    }
    runs = []
    try:
        r = requests.get(endpoint, params=params, timeout=30)
        data = r.json()
        for item in data:
            acc = item.get("run_accession", "")
            ftp_str = item.get("fastq_ftp", "")
            if acc and ftp_str:
                ftp_url = "https://" + ftp_str.split(";")[0]
                runs.append({
                    "class_id": cancer_item["class_id"],
                    "class_name": cancer_item["name"],
                    "accession": acc,
                    "fastq_url": ftp_url
                })
    except Exception as e:
        print(f"   [Error] ENA search for {cancer_item['name']}: {e}")
    return runs

def stream_and_filter_sra_sample(fastq_url, output_fasta, max_reads=100000):
    k = 17
    retained_reads = 0
    try:
        r = requests.get(fastq_url, stream=True, timeout=60)
        with open(output_fasta, "w", encoding="utf-8") as out_f:
            with gzip.open(r.raw, "rt", errors="ignore") as gz:
                header = ""
                for idx, line in enumerate(gz):
                    if idx % 4 == 0:
                        header = line.strip().replace("@", ">")
                    elif idx % 4 == 1:
                        r_seq = line.strip().upper()
                        if len(r_seq) >= k:
                            # Sliding k-mer test
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
        return 0

def process_single_patient(sample_item, sample_idx, total_samples):
    cid = sample_item["class_id"]
    cname_clean = sample_item["class_name"].replace(" ", "_")
    acc = sample_item["accession"]
    fastq_url = sample_item["fastq_url"]
    
    class_subfolder = os.path.join(PATIENT_VCF_DIR, f"class_{cid:02d}_{cname_clean}")
    os.makedirs(class_subfolder, exist_ok=True)
    
    vcf_filename = f"class_{cid:02d}_sample_{sample_idx:03d}_{acc}.vcf"
    vcf_path = os.path.join(class_subfolder, vcf_filename)
    temp_fasta = os.path.join(class_subfolder, f"temp_{acc}.fasta")
    
    # Checkpoint: skip if valid VCF exists
    if os.path.exists(vcf_path) and os.path.getsize(vcf_path) > 2000:
        return {"acc": acc, "class_id": cid, "status": "SKIPPED", "vcf": vcf_path}
        
    print(f"[{sample_idx:03d}/{total_samples:03d}] [Class {cid:02d}: {sample_item['class_name']}] Streaming {acc}...")
    retained_cnt = stream_and_filter_sra_sample(fastq_url, temp_fasta)
    file_size_mb = os.path.getsize(temp_fasta) / (1024 * 1024) if os.path.exists(temp_fasta) else 0
    
    # Call variants
    try:
        vars_cnt = process_reads_and_call_variants(REF_PANEL, temp_fasta, vcf_path, min_dp=2, min_vaf=0.15)
        vcf_size_kb = os.path.getsize(vcf_path) / 1024 if os.path.exists(vcf_path) else 0
        print(f"   [SUCCESS] {vcf_filename} ({vcf_size_kb:.1f} KB, {vars_cnt:,} variants)")
        status = "SUCCESS"
    except Exception as e:
        status = f"FAILED ({e})"
        
    # Delete temp raw reads immediately
    if os.path.exists(temp_fasta):
        os.remove(temp_fasta)
        
    gc.collect()
    return {"acc": acc, "class_id": cid, "status": status, "vcf": vcf_path}

def build_master_dataset():
    print("\n" + "=" * 75)
    print("COMPILING MULTI-CLASS MASTER DEEP LEARNING DATASET CSV")
    print("=" * 75)
    
    cancers = load_taxonomy()
    tax_map = {c["class_id"]: c["name"] for c in cancers}
    
    all_dfs = []
    for c in cancers:
        cid = c["class_id"]
        cname = c["name"]
        cname_clean = cname.replace(" ", "_")
        class_folder = os.path.join(PATIENT_VCF_DIR, f"class_{cid:02d}_{cname_clean}")
        
        vcf_files = sorted(glob.glob(os.path.join(class_folder, "*.vcf")))
        print(f"[Class {cid:02d}: {cname:<25}] Found {len(vcf_files)} VCF files.")
        
        for v_path in vcf_files:
            v_name = os.path.basename(v_path)
            temp_csv = v_path.replace(".vcf", "_encoded.csv")
            temp_schema = v_path.replace(".vcf", "_schema.json")
            
            try:
                df_sample = build_dl_dataset(v_path, REF_PANEL, temp_csv, temp_schema)
                if df_sample is not None and not df_sample.empty:
                    df_sample["patient_vcf"] = v_name
                    df_sample["cancer_class_id"] = cid
                    df_sample["cancer_class_name"] = cname
                    all_dfs.append(df_sample)
            except Exception as e:
                print(f"   Error parsing {v_name}: {e}")
                
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(MASTER_CSV, index=False)
        
        schema = {
            "num_samples": len(final_df),
            "num_features": len(final_df.columns),
            "num_classes": len(cancers),
            "classes": tax_map,
            "target_column": "cancer_class_id",
            "target_name_column": "cancer_class_name",
            "class_distribution": final_df["cancer_class_name"].value_counts().to_dict()
        }
        with open(MASTER_SCHEMA, "w") as f:
            json.dump(schema, f, indent=2)
            
        print("\n" + "=" * 75)
        print("SUCCESS: Multi-Class Master Dataset Generated!")
        print(f"Output File:    {MASTER_CSV}")
        print(f"Total Rows:     {len(final_df):,}")
        print(f"Total Columns:  {final_df.shape[1]}")
        print("=" * 75)
        return final_df
    return None

def run_pipeline(samples_per_cancer=10):
    index_reference()
    cancers = load_taxonomy()
    
    print("=" * 75)
    print(f"DISCOVERING SRA RUNS ACROSS 11 CANCER CLASSES ({samples_per_cancer} Samples / Class)")
    print("=" * 75)
    
    all_runs = []
    for c in cancers:
        runs = fetch_sra_runs_for_cancer(c, limit=samples_per_cancer)
        print(f"[Class {c['class_id']:02d}: {c['name']:<25}] Discovered {len(runs)} ENA runs.")
        all_runs.extend(runs)
        
    total = len(all_runs)
    print(f"\nTotal Multi-Class Cohort: {total} Patient Runs to Process.")
    
    start_time = time.time()
    completed = 0
    skipped = 0
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_patient, item, idx + 1, total): item
            for idx, item in enumerate(all_runs)
        }
        
        for future in as_completed(futures):
            res = future.result()
            if res["status"] == "SKIPPED":
                skipped += 1
                completed += 1
            elif res["status"] == "SUCCESS":
                completed += 1
            
            new_p = completed - skipped
            if new_p > 0:
                elapsed = time.time() - start_time
                avg_sec = elapsed / new_p
                rem_hrs = (avg_sec * (total - completed)) / 3600.0
                print(f"   [Progress] {completed}/{total} done (Resumed: {skipped}) | ETA: {rem_hrs:.2f} hrs")

    # Build Master CSV
    build_master_dataset()

if __name__ == "__main__":
    count = 10
    if len(sys.argv) > 1:
        count = int(sys.argv[1])
    run_pipeline(samples_per_cancer=count)
