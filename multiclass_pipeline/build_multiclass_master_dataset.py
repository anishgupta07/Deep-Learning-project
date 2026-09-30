#!/usr/bin/env python3
"""
Multi-Class Master Deep Learning Dataset Builder
================================================
Aggregates all 11-cancer patient VCF files into a unified multi-class 
Deep Learning dataset matrix with 21-bp one-hot sequence encodings.
"""

import sys
import os
import glob
import json
import pandas as pd

# Add parent directory for feature extraction engine
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from build_dl_dataset import build_dl_dataset

TAXONOMY_PATH = os.path.join(os.path.dirname(__file__), "cancer_taxonomy.json")

def load_taxonomy_map():
    with open(TAXONOMY_PATH, "r") as f:
        cancers = json.load(f)["cancers"]
    return {c["class_id"]: c["name"] for c in cancers}

def build_multiclass_master_dataset(vcf_dir="vcf_output_multiclass", ref_fasta="multiclass_cancer_reference_panel.fna", output_csv="multiclass_cancer_master_dataset.csv", output_schema="multiclass_dataset_schema.json"):
    print("=" * 70)
    print("BUILDING MULTI-CLASS DEEP LEARNING MASTER DATASET (11 CANCER TYPES)")
    print(f"VCF Directory:   {vcf_dir}/")
    print(f"Reference Panel: {ref_fasta}")
    print(f"Output CSV:      {output_csv}")
    print("=" * 70)
    
    tax_map = load_taxonomy_map()
    vcf_files = sorted(glob.glob(os.path.join(vcf_dir, "*.vcf")))
    print(f"Found {len(vcf_files):,} Multi-Class Patient VCF files.")
    
    all_dfs = []
    for idx, v_path in enumerate(vcf_files, 1):
        v_name = os.path.basename(v_path)
        
        # Parse class ID from filename, e.g. class_00_Oral_Cavity_Carcinoma_sample_0001_SRR...
        try:
            cid = int(v_name.split("_")[1])
            cname = tax_map.get(cid, "Unknown")
        except Exception:
            cid = 0
            cname = "Oral Cavity Carcinoma"
            
        temp_csv = v_path.replace(".vcf", "_encoded.csv")
        temp_schema = v_path.replace(".vcf", "_schema.json")
        
        try:
            df_sample = build_dl_dataset(v_path, ref_fasta, temp_csv, temp_schema)
            if df_sample is not None and not df_sample.empty:
                df_sample["patient_vcf"] = v_name
                df_sample["cancer_class_id"] = cid
                df_sample["cancer_class_name"] = cname
                all_dfs.append(df_sample)
                print(f"[{idx:04d}/{len(vcf_files):04d}] [Class {cid:02d}: {cname:<25}] Extracted {len(df_sample):,} variants")
        except Exception as e:
            print(f"[{idx:04d}/{len(vcf_files):04d}] Error for {v_name}: {e}")
            
    if all_dfs:
        master_df = pd.concat(all_dfs, ignore_index=True)
        master_df.to_csv(output_csv, index=False)
        
        # Save Multi-Class Schema JSON
        schema = {
            "num_samples": len(master_df),
            "num_features": len(master_df.columns),
            "num_classes": 11,
            "classes": tax_map,
            "target_column": "cancer_class_id",
            "target_name_column": "cancer_class_name",
            "tensor_features": {
                "one_hot_ref_dim": 84,
                "one_hot_alt_dim": 84,
                "total_sequence_tensor_dim": 168
            },
            "class_distribution": master_df["cancer_class_name"].value_counts().to_dict()
        }
        with open(output_schema, "w") as f:
            json.dump(schema, f, indent=2)
            
        print("\n" + "=" * 70)
        print("SUCCESS: Multi-Class Master Dataset Generated!")
        print(f"Total Rows:       {len(master_df):,}")
        print(f"Total Columns:    {master_df.shape[1]}")
        print(f"Target Classes:   11 Cancer Types")
        print(f"Saved CSV Matrix: {output_csv}")
        print(f"Saved Schema:     {output_schema}")
        print("=" * 70)
        return master_df
    else:
        print("No variant data found across VCF files.")
        return None

if __name__ == "__main__":
    v_dir = "vcf_output_multiclass"
    r_panel = "multiclass_cancer_reference_panel.fna"
    out_file = "multiclass_cancer_master_dataset.csv"
    
    if len(sys.argv) > 1:
        v_dir = sys.argv[1]
    if len(sys.argv) > 2:
        r_panel = sys.argv[2]
    if len(sys.argv) > 3:
        out_file = sys.argv[3]
        
    build_multiclass_master_dataset(vcf_dir=v_dir, ref_fasta=r_panel, output_csv=out_file)
