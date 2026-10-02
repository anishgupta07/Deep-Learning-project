import bio_chemistry_engine as bce
# ==============================================================================
# Pan-Cancer SRA-to-VCF Streaming Pipeline Engine
# Multi-Class Germline Pan-Cancer Risk Engine (11 Cancer Classes, 130 Genes)
# Output CSV matches exact user specification:
# Columns (201 total = 10 details + 9 metrics/labels + 14 RDKit/Evolution/MI + 168 one-hot = 10 details + 9 metrics/labels + 84 ref one-hot + 84 alt one-hot):
#   variant_id, chrom, pos, ref, alt, mutation, is_transition, trinucleotide,
#   ref_seq_21, alt_seq_21, gc_content, dp, af, ro, ao, qual, gt_code,
#   label_high, label_som,
#   one_hot_ref_0_A, one_hot_ref_0_C, one_hot_ref_0_G, one_hot_ref_0_T, ...
#   ... up to one_hot_ref_20_T (84 one-hot columns across 21 positions)
# ==============================================================================

import os
import sys
import time
import json
import argparse
import subprocess
import numpy as np
import pandas as pd
from Bio import SeqIO

BASE_DIR = r"D:\DL"
REF_PANEL_WIN = os.path.join(BASE_DIR, "unified_128_gene_reference_panel.fna")
COHORT_CSV = os.path.join(BASE_DIR, "master_1100_patient_cohorts.csv")

CLASS_DIRS = {
    0: "class_00_oral_cavity_carcinoma",
    1: "class_01_lung_cancer",
    2: "class_02_female_breast_cancer",
    3: "class_03_colorectal_cancer",
    4: "class_04_prostate_cancer",
    5: "class_05_stomach_cancer",
    6: "class_06_thyroid_cancer",
    7: "class_07_liver_cancer",
    8: "class_08_bladder_cancer",
    9: "class_09_cervical_cancer",
    10: "class_10_non_hodgkin_lymphoma"
}

TRANSITIONS = {('A', 'G'), ('G', 'A'), ('C', 'T'), ('T', 'C')}

def win_to_wsl(win_path):
    win_path = os.path.abspath(win_path)
    drive = win_path[0].lower()
    rest = win_path[2:].replace("\\", "/")
    return f"/mnt/{drive}{rest}"

def download_sra_fasta(accession, dest_path):
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 10000:
        print(f"[*] [1/5] FASTA already present for {accession} ({os.path.getsize(dest_path)/(1024*1024):.2f} MB). Skipping download.")
        return os.path.getsize(dest_path)
    url = f"https://trace.ncbi.nlm.nih.gov/Traces/sra-reads-be/fasta?acc={accession}"
    print(f"[*] [1/5] Downloading filtered FASTA for {accession} via curl.exe...")
    t0 = time.time()
    
    cmd = [
        "curl.exe", "-L",
        "--retry", "5",
        "--retry-delay", "2",
        "-s",
        "-w", "%{size_download}",
        "-o", dest_path,
        url
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(dest_path) or os.path.getsize(dest_path) == 0:
        print(f"[-] Download failed for {accession}: {res.stderr}")
        return 0
        
    elapsed = time.time() - t0
    total_bytes = os.path.getsize(dest_path)
    mb = total_bytes / (1024 * 1024)
    speed = mb / elapsed if elapsed > 0 else 0
    print(f"[+] [1/5] Download Complete: {mb:.2f} MB in {elapsed:.1f}s ({speed:.2f} MB/s)")
    return total_bytes

def align_and_call_variants(accession, fasta_path, class_dir):
    wsl_ref = win_to_wsl(REF_PANEL_WIN)
    wsl_fasta = win_to_wsl(fasta_path)
    bam_path = os.path.join(class_dir, f"{accession}.bam")
    vcf_path = os.path.join(class_dir, f"{accession}.vcf")
    wsl_bam = win_to_wsl(bam_path)
    wsl_vcf = win_to_wsl(vcf_path)
    
    print(f"[*] [2/5] Aligning reads & calling variants via WSL...")
    t0 = time.time()
    cmd = (
        f"minimap2 -ax sr -t 4 {wsl_ref} {wsl_fasta} 2>/dev/null | "
        f"samtools view -b -F 4 - | "
        f"samtools sort -@ 4 -o {wsl_bam} && "
        f"samtools index {wsl_bam} && "
        f"bcftools mpileup -Ou -f {wsl_ref} {wsl_bam} 2>/dev/null | "
        f"bcftools call -mv -Ov -o {wsl_vcf}"
    )
    
    res = subprocess.run(["wsl", "bash", "-c", cmd], capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[-] Error during alignment or variant calling: {res.stderr}")
        return None, res.stderr
        
    elapsed = time.time() - t0
    print(f"[+] [2/5] Alignment & Variant Calling Complete in {elapsed:.1f}s")
    
    # Clean up intermediate BAM & BAI
    for ext in ["", ".bai"]:
        p = bam_path + ext
        if os.path.exists(p):
            os.remove(p)
            
    return vcf_path, None

def generate_exact_schema_csv_and_tensors(accession, vcf_path, ref_path, class_dir):
    print(f"[*] [3/5] Extracting exact 103-column CSV schema (trinucleotide, 21bp windows, metrics, one-hot)...")
    ref_dict = SeqIO.to_dict(SeqIO.parse(ref_path, "fasta"))
    
    rows = []
    with open(vcf_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#"): continue
            parts = line.strip().split("\t")
            chrom = parts[0]
            pos = int(parts[1])
            ref = parts[3].upper()
            alt = parts[4].upper()
            qual = float(parts[5])
            info = parts[7]
            format_keys = parts[8].split(":") if len(parts) > 8 else []
            sample_vals = parts[9].split(":") if len(parts) > 9 else []
            fmt = dict(zip(format_keys, sample_vals))
            info_dict = dict(item.split("=") for item in info.split(";") if "=" in item)
            
            # Keep only single-base SNVs
            if len(ref) != 1 or len(alt) != 1 or chrom not in ref_dict:
                continue
            seq = str(ref_dict[chrom].seq)
            if pos < 11 or pos + 10 > len(seq):
                continue
                
            upstream = seq[pos - 11 : pos - 1].upper()
            downstream = seq[pos : pos + 10].upper()
            ref_seq_21 = upstream + ref + downstream
            alt_seq_21 = upstream + alt + downstream
            
            variant_id = f"{chrom}_{pos}_{ref}_{alt}"
            mutation = "SNV"
            is_trans = 1 if (ref, alt) in TRANSITIONS else 0
            trinuc = f"{upstream[-1]}[{ref}>{alt}]{downstream[0]}"
            gc_count = ref_seq_21.count("G") + ref_seq_21.count("C")
            gc_content = round(gc_count / 21.0, 4)
            
            dp = int(info_dict.get("DP", 0))
            if dp == 0 and "DP" in fmt:
                try: dp = int(fmt["DP"])
                except: dp = 1
            if dp == 0: dp = 1
            
            gt = fmt.get("GT", "1/1")
            af_str = info_dict.get("AF")
            if gt in ["1/1", "1|1"]:
                gt_code = 2
                af = 1.0
                ro = 0
                ao = dp
            elif gt in ["0/1", "1/0", "0|1", "1|0"]:
                gt_code = 1
                af = 0.5
                ro = max(1, int(round(dp * 0.5)))
                ao = max(1, dp - ro)
            else:
                gt_code = 2
                af = float(af_str.split(",")[0]) if af_str else 1.0
                ro = int(round(dp * (1 - af)))
                ao = dp - ro
                
            row_dict = {
                "variant_id": variant_id,
                "chrom": chrom,
                "pos": pos,
                "ref": ref,
                "alt": alt,
                "mutation": mutation,
                "is_transition": is_trans,
                "trinucleotide": trinuc,
                "ref_seq_21": ref_seq_21,
                "alt_seq_21": alt_seq_21,
                "gc_content": gc_content,
                "dp": dp,
                "af": af,
                "ro": ro,
                "ao": ao,
                "qual": round(qual, 1),
                "gt_code": gt_code,
                "label_high": 0,
                "label_som": 0
            }
            # Compute RDKit Amino Acid Chemistry, dN/dS & Mutual Information
            chem_feats = bce.compute_mutation_features(row_dict)
            row_dict.update(chem_feats)
            
            for i, ch in enumerate(ref_seq_21):
                row_dict[f"one_hot_ref_{i}_A"] = 1 if ch == "A" else 0
                row_dict[f"one_hot_ref_{i}_C"] = 1 if ch == "C" else 0
                row_dict[f"one_hot_ref_{i}_G"] = 1 if ch == "G" else 0
                row_dict[f"one_hot_ref_{i}_T"] = 1 if ch == "T" else 0

            for i, ch in enumerate(alt_seq_21):
                row_dict[f"one_hot_alt_{i}_A"] = 1 if ch == "A" else 0
                row_dict[f"one_hot_alt_{i}_C"] = 1 if ch == "C" else 0
                row_dict[f"one_hot_alt_{i}_G"] = 1 if ch == "G" else 0
                row_dict[f"one_hot_alt_{i}_T"] = 1 if ch == "T" else 0
                
            rows.append(row_dict)
            
    df = pd.DataFrame(rows)
    csv_path = os.path.join(class_dir, f"{accession}_mutations_21bp.csv")
    try:
        df.to_csv(csv_path, index=False)
    except PermissionError:
        csv_path = os.path.join(class_dir, f"{accession}_mutations_21bp_v2.csv")
        df.to_csv(csv_path, index=False)
        
    print(f"[+] [3/5] Saved Exact Schema CSV: {csv_path} ({len(df):,} rows x {len(df.columns)} columns).")
    
    # Only saving exact schema CSV per user requirement (skipping .npy tensor files)
    pass
        
    return len(rows)

def process_single_sra(accession, class_id, keep_fasta=False):
    class_folder = CLASS_DIRS.get(class_id)
    if not class_folder:
        raise ValueError(f"Invalid class_id: {class_id}")
        
    class_dir = os.path.join(BASE_DIR, class_folder)
    os.makedirs(class_dir, exist_ok=True)
    
    vcf_path = os.path.join(class_dir, f"{accession}.vcf")
    csv_path = os.path.join(class_dir, f"{accession}_mutations_21bp.csv")
    
    if os.path.exists(csv_path) and os.path.getsize(csv_path) > 1000:
        print(f"[*] Skipping {accession}: already processed ({os.path.getsize(csv_path):,} bytes).")
        return True
        
    print("=" * 75)
    print(f">>> STARTING PIPELINE: Accession: {accession} | Class {class_id}: {class_folder}")
    print("=" * 75)
    
    fasta_path = os.path.join(class_dir, f"{accession}.fasta")
    
    # 1. Download
    download_size = download_sra_fasta(accession, fasta_path)
    if download_size == 0:
        return False
    
    # 2. Align & Call Variants
    out_vcf, err = align_and_call_variants(accession, fasta_path, class_dir)
    if err or not out_vcf or not os.path.exists(out_vcf):
        print(f"[-] VCF generation failed for {accession}")
        return False
        
    # 3. Exact Schema CSV & Tensors
    variants = generate_exact_schema_csv_and_tensors(accession, vcf_path, REF_PANEL_WIN, class_dir)
    
    # 4. Delete SRA FASTA and intermediate VCF
    if not keep_fasta:
        print(f"[*] [4/5] Deleting raw SRA FASTA ({download_size / (1024*1024):.2f} MB)...")
        if os.path.exists(fasta_path):
            os.remove(fasta_path)
        print(f"[+] [4/5] SRA FASTA deleted.")
    if os.path.exists(out_vcf):
        try:
            os.remove(out_vcf)
            print(f"[+] [4/5] Intermediate VCF deleted (CSV is master dataset).")
        except Exception as e:
            print(f"[!] Warning deleting VCF: {e}")
        
    print(f"[+] [5/5] Patient {accession} complete ({variants:,} mutations in exact schema CSV).")
    print("=" * 75)
    return True

def run_cohort(class_id, limit=100):
    class_folder = CLASS_DIRS.get(class_id)
    class_dir = os.path.join(BASE_DIR, class_folder)
    os.makedirs(class_dir, exist_ok=True)
    
    import glob
    class_cohort_files = glob.glob(os.path.join(class_dir, "*cohort*.csv"))
    if class_cohort_files:
        df = pd.read_csv(class_cohort_files[0])
        print(f"[*] Using class cohort file: {class_cohort_files[0]}")
    elif os.path.exists(COHORT_CSV):
        df = pd.read_csv(COHORT_CSV)
        df = df[df["Cancer_Class_ID"] == class_id]
        print(f"[*] Using master cohort CSV: {COHORT_CSV}")
    else:
        print("[-] No cohort file found.")
        return
        
    sub = df.head(limit)
    print(f"[*] Processing {len(sub)} samples for Class {class_id} ({class_folder})...")
    
    progress_file = os.path.join(class_dir, "cohort_progress.json")
    progress = {}
    if os.path.exists(progress_file):
        try:
            with open(progress_file, "r") as f:
                progress = json.load(f)
        except:
            progress = {}
            
    success_count = 0
    t_start = time.time()
    
    for idx, row in sub.iterrows():
        acc = row["SRA_Run_Accession"]
        p_idx = row.get("Patient_Index", idx + 1)
        print(f"\n===========================================================================")
        print(f">>> Cohort Progress: Patient {p_idx}/{len(sub)} [Accession: {acc}]")
        print(f"===========================================================================")
        
        ok = process_single_sra(acc, class_id)
        if ok:
            success_count += 1
            progress[acc] = {
                "patient_index": int(p_idx),
                "status": "COMPLETED",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            with open(progress_file, "w") as f:
                json.dump(progress, f, indent=2)
                
    elapsed_total = time.time() - t_start
    print("\n" + "#" * 75)
    print(f"[COHORT COMPLETE] Class {class_id} finished: {success_count}/{len(sub)} in {elapsed_total/60:.1f} min!")
    print("#" * 75)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pan-Cancer SRA to VCF Pipeline Engine")
    parser.add_argument("--accession", help="Single SRA Run Accession (e.g. SRR40755076)")
    parser.add_argument("--class-id", type=int, required=True, help="Cancer Class ID (0-10)")
    parser.add_argument("--cohort", action="store_true", help="Process full cohort for class")
    parser.add_argument("--limit", type=int, default=100, help="Max patients to process in cohort mode")
    parser.add_argument("--keep-fasta", action="store_true", help="Keep raw FASTA file instead of deleting")
    args = parser.parse_args()
    
    if args.cohort:
        run_cohort(args.class_id, limit=args.limit)
    elif args.accession:
        success = process_single_sra(args.accession, args.class_id, keep_fasta=args.keep_fasta)
        sys.exit(0 if success else 1)
    else:
        print("[-] Please specify either --accession or --cohort")
