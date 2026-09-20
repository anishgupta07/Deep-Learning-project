#!/usr/bin/env python3
"""
One-Sample Automated Pipeline for 2 Genes & 2 SRA Files
========================================================
Combines uploaded gene FASTA files, runs variant calling for 2 patient SRA files,
generates 2 individual VCFs, and builds the 21-bp one-hot encoded CSV dataset.
"""

import sys
import os
import glob
import pandas as pd

# Add parent directory to path to import core engines
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from generate_vcf import process_reads_and_call_variants
from build_dl_dataset import build_dl_dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
VCF_DIR = os.path.join(BASE_DIR, "vcf_output")
COMBINED_REF = os.path.join(BASE_DIR, "combined_reference_genes.fna")
OUTPUT_CSV = os.path.join(BASE_DIR, "one_sample_dataset.csv")

def combine_gene_fasta_files():
    """Combines all gene reference FASTA files in raw_data/ into combined_reference_genes.fna"""
    print(f"1. Scanning {RAW_DIR} for Gene Reference FASTA files...")
    all_files = glob.glob(os.path.join(RAW_DIR, "*.*"))
    
    gene_files = [
        f for f in all_files 
        if ("gene" in os.path.basename(f).lower() or f.endswith(".fna") or "ref" in os.path.basename(f).lower())
        and not ("sra" in os.path.basename(f).lower() or "patient" in os.path.basename(f).lower() or "person" in os.path.basename(f).lower() or f.endswith(".md"))
    ]
    
    if not gene_files:
        gene_files = [f for f in all_files if f.endswith((".fasta", ".fna", ".fa")) and not ("sra" in os.path.basename(f).lower() or "patient" in os.path.basename(f).lower() or "srr" in os.path.basename(f).lower() or f.endswith(".md"))]
        
    print(f"   Found {len(gene_files)} Gene Reference File(s): {[os.path.basename(g) for g in gene_files]}")
    
    with open(COMBINED_REF, "w", encoding="utf-8") as out_f:
        for g_file in gene_files:
            with open(g_file, "r", encoding="utf-8", errors="ignore") as in_f:
                out_f.write(in_f.read().strip() + "\n\n")
                
    ref_size_kb = os.path.getsize(COMBINED_REF) / 1024
    print(f"   SUCCESS: Combined reference saved to {COMBINED_REF} ({ref_size_kb:.1f} KB)")
    return COMBINED_REF

def process_patient_sra_files():
    """Processes patient SRA files against combined reference FASTA and generates VCFs"""
    os.makedirs(VCF_DIR, exist_ok=True)
    all_files = glob.glob(os.path.join(RAW_DIR, "*.*"))
    
    sra_files = [
        f for f in all_files 
        if ("sra" in os.path.basename(f).lower() or "patient" in os.path.basename(f).lower() or "srr" in os.path.basename(f).lower() or "person" in os.path.basename(f).lower() or "read" in os.path.basename(f).lower())
        and not f.endswith(".md")
    ]
    
    print(f"\n2. Found {len(sra_files)} Patient SRA File(s) to process:")
    for s in sra_files:
        print(f"   - {os.path.basename(s)}")
        
    vcf_paths = []
    for idx, sra_path in enumerate(sorted(sra_files), 1):
        sample_name = os.path.basename(sra_path).split(".")[0]
        vcf_filename = f"person_{idx:02d}_{sample_name}.vcf"
        vcf_path = os.path.join(VCF_DIR, vcf_filename)
        
        print(f"\n   --- Running Variant Calling for Sample {idx}/{len(sra_files)}: {sample_name} ---")
        print(f"       Reference: {os.path.basename(COMBINED_REF)}")
        print(f"       Patient:   {os.path.basename(sra_path)}")
        print(f"       Output:    {os.path.basename(vcf_path)}")
        
        try:
            process_reads_and_call_variants(COMBINED_REF, sra_path, vcf_path, min_dp=2, min_vaf=0.15)
            vcf_size_kb = os.path.getsize(vcf_path) / 1024 if os.path.exists(vcf_path) else 0
            print(f"       [SUCCESS] VCF written: {vcf_filename} ({vcf_size_kb:.1f} KB)")
            vcf_paths.append(vcf_path)
        except Exception as err:
            print(f"       [Error] Failed for {sample_name}: {err}")
            
    return vcf_paths

def build_combined_csv_dataset(vcf_paths):
    """Converts individual VCFs into 21-bp one-hot encoded CSV datasets and aggregates them"""
    print(f"\n3. Converting VCF files into 21-bp One-Hot Encoded CSV dataset...")
    all_dfs = []
    
    for v_path in vcf_paths:
        sample_name = os.path.basename(v_path).replace(".vcf", "")
        temp_csv = v_path.replace(".vcf", "_encoded.csv")
        temp_schema = v_path.replace(".vcf", "_schema.json")
        
        try:
            # build_dl_dataset(vcf_path, ref_fasta_path, output_csv_path, output_schema_path)
            df_sample = build_dl_dataset(v_path, COMBINED_REF, temp_csv, temp_schema)
            if df_sample is not None and not df_sample.empty:
                df_sample["patient_id"] = sample_name
                all_dfs.append(df_sample)
                print(f"   - Extracted {len(df_sample):,} variant rows for {sample_name}")
            else:
                print(f"   - No variants in {sample_name}")
        except Exception as err:
            print(f"   - [Error] CSV conversion failed for {sample_name}: {err}")
            
    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        final_df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n" + "=" * 65)
        print(f"SUCCESS: Combined One-Sample Dataset Saved!")
        print(f"Output File:       {OUTPUT_CSV}")
        print(f"Total Rows:        {len(final_df):,}")
        print(f"Total Columns:     {final_df.shape[1]}")
        print("=" * 65)
        return OUTPUT_CSV
    else:
        print("   [Warning] No datasets generated.")
        return None

if __name__ == "__main__":
    print("=" * 65)
    print("AUTOMATED ONE-SAMPLE PIPELINE (2 GENES & 2 SRA SAMPLES)")
    print("=" * 65)
    ref_path = combine_gene_fasta_files()
    vcfs = process_patient_sra_files()
    if vcfs:
        build_combined_csv_dataset(vcfs)
