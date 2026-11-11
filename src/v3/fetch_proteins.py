"""Download and cache canonical human protein sequences for the 130 panel genes from UniProt.

Matches canonical sequences against MANE/UniProt lengths in data/protein_lengths.json.
Saves canonical sequences to data/panel_proteins.fasta.
Validates sequence residue against ClinVar ref_aa at aa_pos.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PANEL_JSON = DATA_DIR / "panel.json"
LENGTHS_JSON = DATA_DIR / "protein_lengths.json"
OUTPUT_FASTA = DATA_DIR / "panel_proteins.fasta"
CLINVAR_CSV = DATA_DIR / "clinvar_labelled_v3.csv"


def parse_fasta(fasta_text: str) -> list[tuple[str, str, str, str]]:
    """Parse fasta text into list of (header, accession, gene_name, sequence)."""
    records = []
    curr_header = ""
    curr_seq = []
    
    for line in fasta_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if curr_header:
                records.append((curr_header, "".join(curr_seq)))
            curr_header = line
            curr_seq = []
        else:
            curr_seq.append(line)
    if curr_header:
        records.append((curr_header, "".join(curr_seq)))

    parsed = []
    for header, seq in records:
        # Accession: >sp|P38398|...
        m_acc = re.match(r">sp\|([^|]+)\|", header)
        acc = m_acc.group(1) if m_acc else ""
        # Gene: GN=BRCA1
        m_gn = re.search(r"GN=([A-Za-z0-9_-]+)", header)
        gn = m_gn.group(1) if m_gn else ""
        parsed.append((header, acc, gn, seq))
    return parsed


def fetch_uniprot_batch(genes: list[str]) -> str:
    """Fetch fasta for a list of gene symbols from UniProt."""
    or_clause = " OR ".join([f"gene:{g}" for g in genes])
    q = f"({or_clause}) AND organism_id:9606 AND reviewed:true"
    params = urllib.parse.urlencode({"query": q, "format": "fasta", "size": 250})
    url = f"https://rest.uniprot.org/uniprotkb/search?{params}"
    
    req = urllib.request.Request(url, headers={"User-Agent": "GenomeDL/3.0 (academic research)"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def fetch_all_panel_sequences(force_download: bool = False, allow_partial: bool = False) -> dict[str, str]:
    """Fetch canonical protein sequences for all 130 panel genes.
    
    Returns:
        dict: gene_symbol -> amino_acid_sequence
    """
    panel = json.load(open(PANEL_JSON, "r", encoding="utf-8"))
    lengths = json.load(open(LENGTHS_JSON, "r", encoding="utf-8"))
    genes = sorted(list(panel.keys()))

    if OUTPUT_FASTA.exists() and not force_download:
        print(f"Loading existing FASTA from {OUTPUT_FASTA}...")
        gene_seqs = {}
        curr_gene = ""
        curr_seq = []
        with open(OUTPUT_FASTA, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith(">"):
                    if curr_gene:
                        gene_seqs[curr_gene] = "".join(curr_seq)
                    curr_gene = line[1:].split()[0]
                    curr_seq = []
                else:
                    curr_seq.append(line)
            if curr_gene:
                gene_seqs[curr_gene] = "".join(curr_seq)
        missing_cached = [g for g in genes if g not in gene_seqs]
        if not missing_cached:
            print(f"Loaded all {len(gene_seqs)} sequences from cache.")
            return gene_seqs

    print(f"Fetching sequences for {len(genes)} genes from UniProt...")
    batch_size = 20
    all_records = []
    
    for i in range(0, len(genes), batch_size):
        batch = genes[i : i + batch_size]
        print(f"  Fetching batch {i // batch_size + 1}/{(len(genes) + batch_size - 1) // batch_size} ({len(batch)} genes)...")
        try:
            fasta_txt = fetch_uniprot_batch(batch)
            parsed = parse_fasta(fasta_txt)
            all_records.extend(parsed)
        except Exception as e:
            print(f"  Batch failed ({e}), falling back to individual queries for batch...")
            for g in batch:
                try:
                    q = f"gene_exact:{g} AND organism_id:9606 AND reviewed:true"
                    params = urllib.parse.urlencode({"query": q, "format": "fasta", "size": 10})
                    url = f"https://rest.uniprot.org/uniprotkb/search?{params}"
                    req = urllib.request.Request(url, headers={"User-Agent": "GenomeDL/3.0"})
                    with urllib.request.urlopen(req, timeout=15) as resp:
                        all_records.extend(parse_fasta(resp.read().decode("utf-8")))
                except Exception as ind_e:
                    print(f"    Failed {g}: {ind_e}")

    # Map records to genes
    gene_to_seq = {}
    records_by_gn = {}
    for header, acc, gn, seq in all_records:
        gn_upper = gn.upper()
        if gn_upper not in records_by_gn:
            records_by_gn[gn_upper] = []
        records_by_gn[gn_upper].append((header, acc, seq))

    # Match each panel gene
    missing_genes = []
    for g in genes:
        exp_len = lengths.get(g)
        recs = records_by_gn.get(g.upper(), [])
        if not recs:
            # Try searching by gene name in header
            for header, acc, gn, seq in all_records:
                if f"GN={g} " in header or header.endswith(f"GN={g}"):
                    recs.append((header, acc, seq))
        
        if recs:
            # Find best match by length
            best_seq = recs[0][2]
            for h, a, s in recs:
                if len(s) == exp_len:
                    best_seq = s
                    break
            gene_to_seq[g] = best_seq
        else:
            missing_genes.append(g)

    # If any gene missing, query specifically
    if missing_genes:
        print(f"Retrying {len(missing_genes)} missing genes: {missing_genes}")
        for g in missing_genes:
            try:
                url = f"https://rest.uniprot.org/uniprotkb/search?query=gene:{g}+AND+organism_id:9606+AND+reviewed:true&format=fasta"
                req = urllib.request.Request(url, headers={"User-Agent": "GenomeDL/3.0"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    recs = parse_fasta(resp.read().decode("utf-8"))
                if recs:
                    gene_to_seq[g] = recs[0][3]
                    print(f"  Recovered {g} (length {len(recs[0][3])})")
            except Exception as e:
                print(f"  Could not recover {g}: {e}")

    still_missing = [g for g in genes if g not in gene_to_seq]
    print(f"Successfully retrieved sequences for {len(gene_to_seq)}/{len(genes)} genes.")
    if still_missing and not allow_partial:
        raise RuntimeError(
            f"Incomplete protein panel: failed to retrieve {len(still_missing)}/{len(genes)} sequences: {still_missing}"
        )

    # Save to FASTA
    OUTPUT_FASTA.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FASTA, "w", encoding="utf-8") as f:
        for g in sorted(gene_to_seq.keys()):
            f.write(f">{g} length={len(gene_to_seq[g])}\n")
            seq = gene_to_seq[g]
            for j in range(0, len(seq), 60):
                f.write(seq[j : j + 60] + "\n")
    print(f"Saved FASTA to {OUTPUT_FASTA}")

    return gene_to_seq


def validate_against_clinvar(gene_seqs: dict[str, str]) -> dict:
    """Validate sequence residues against ClinVar missense variants."""
    df = pd.read_csv(CLINVAR_CSV, low_memory=False)
    prot_mask = df["variant_class"].isin(["missense", "stop_lost", "start_lost"])
    prot_df = df[prot_mask].copy()

    total_variants = len(prot_df)
    matched = 0
    mismatched = 0
    out_of_bounds = 0
    missing_gene = 0

    mismatch_details = []

    for idx, row in prot_df.iterrows():
        gene = str(row["gene"])
        ref_aa = str(row["ref_aa"]) if pd.notna(row["ref_aa"]) else ""
        aa_pos = int(row["aa_pos"]) if pd.notna(row["aa_pos"]) else -1

        if gene not in gene_seqs:
            missing_gene += 1
            continue

        seq = gene_seqs[gene]
        if aa_pos < 1 or aa_pos > len(seq):
            out_of_bounds += 1
            continue

        actual_aa = seq[aa_pos - 1]
        if actual_aa == ref_aa:
            matched += 1
        else:
            mismatched += 1
            if len(mismatch_details) < 10:
                mismatch_details.append((gene, row["hgvs_name"], aa_pos, ref_aa, actual_aa))

    match_rate = matched / total_variants if total_variants > 0 else 0.0
    print("\n--- ClinVar Sequence Validation ---")
    print(f"Total protein variants: {total_variants}")
    print(f"Matched residues:       {matched} ({match_rate * 100:.2f}%)")
    print(f"Mismatched residues:    {mismatched} ({mismatched / total_variants * 100:.2f}%)")
    print(f"Out of bounds:          {out_of_bounds}")
    print(f"Missing gene seq:       {missing_gene}")
    if mismatch_details:
        print("Sample mismatches (first 5):")
        for m in mismatch_details[:5]:
            print(f"  {m[0]} {m[1]} pos={m[2]}: expected ref={m[3]}, actual in seq={m[4]}")

    return {
        "total": total_variants,
        "matched": matched,
        "mismatched": mismatched,
        "match_rate": match_rate,
        "out_of_bounds": out_of_bounds,
    }


if __name__ == "__main__":
    gene_seqs = fetch_all_panel_sequences()
    validate_against_clinvar(gene_seqs)
