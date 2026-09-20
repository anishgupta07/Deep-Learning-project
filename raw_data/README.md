# Raw Data Upload Directory for 2 Genes & 2 SRA Files

Please place your raw files here:
- **2 Gene FASTA files** (e.g., `gene1.fna`, `gene2.fna` or combined `genes.fasta`)
- **2 Patient SRA files** (e.g., `patient1.fasta` / `SRR1.fasta`, `patient2.fasta` / `SRR2.fasta`)

Once uploaded, the pipeline will:
1. Combine the 2 gene FASTA files into `combined_reference_genes.fna`.
2. Run variant calling on Patient 1 and Patient 2 against the combined reference.
3. Generate 2 VCF files (`person_01.vcf` and `person_02.vcf`).
4. Convert both VCFs into the encoded 21-bp sequence Deep Learning CSV matrix (`one_sample_dataset.csv`).
