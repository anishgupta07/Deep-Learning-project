from Bio import SeqIO
import os
import sys

def filter_and_annotate_vcf_21bp(vcf_path, ref_path):
    if not os.path.exists(vcf_path):
        print(f"VCF not found: {vcf_path}")
        return False
        
    print(f"Loading reference panel from {ref_path}...")
    ref_dict = SeqIO.to_dict(SeqIO.parse(ref_path, "fasta"))
    print(f"Loaded {len(ref_dict)} contigs.")
    
    tmp_vcf = vcf_path + ".tmp"
    header_lines = []
    data_lines = []
    
    new_headers = [
        '##INFO=<ID=WINDOW21,Number=1,Type=String,Description="21-bp spatial DNA sequence window centered on mutated allele">\n',
        '##INFO=<ID=REF21,Number=1,Type=String,Description="21-bp spatial DNA sequence window centered on reference allele">\n'
    ]
    
    total_raw = 0
    kept_count = 0
    dropped_non_snv = 0
    dropped_boundary = 0
    
    with open(vcf_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("##"):
                header_lines.append(line)
            elif line.startswith("#CHROM"):
                header_lines.extend(new_headers)
                header_lines.append(line)
            else:
                total_raw += 1
                parts = line.strip().split("\t")
                chrom = parts[0]
                pos = int(parts[1])
                ref = parts[3]
                alt = parts[4]
                info = parts[7]
                
                # Filter: Keep ONLY single-base mutations (SNVs) where a 21-bp window exists
                if len(ref) != 1 or len(alt) != 1:
                    dropped_non_snv += 1
                    continue
                if chrom not in ref_dict:
                    continue
                    
                seq = str(ref_dict[chrom].seq)
                if pos < 11 or pos + 10 > len(seq):
                    dropped_boundary += 1
                    continue
                    
                upstream = seq[pos - 11 : pos - 1]
                downstream = seq[pos : pos + 10]
                win_alt = upstream + alt + downstream
                win_ref = upstream + ref + downstream
                
                parts[7] = f"{info};WINDOW21={win_alt};REF21={win_ref}"
                data_lines.append("\t".join(parts) + "\n")
                kept_count += 1
                
    with open(tmp_vcf, "w", encoding="utf-8") as f:
        f.writelines(header_lines)
        f.writelines(data_lines)
        
    os.replace(tmp_vcf, vcf_path)
    print(f"Successfully processed {vcf_path}:")
    print(f"  - Total raw calls: {total_raw:,}")
    print(f"  - Kept with 21-bp window: {kept_count:,}")
    print(f"  - Dropped non-SNVs: {dropped_non_snv:,}")
    print(f"  - Dropped boundary: {dropped_boundary:,}")
    return True

if __name__ == "__main__":
    vcf = sys.argv[1] if len(sys.argv) > 1 else r"D:\DL\class_01_lung_cancer\SRR40755076.vcf"
    ref = sys.argv[2] if len(sys.argv) > 2 else r"D:\DL\unified_128_gene_reference_panel.fna"
    filter_and_annotate_vcf_21bp(vcf, ref)
