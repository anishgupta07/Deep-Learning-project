#!/usr/bin/env bash
set -euo pipefail

# GenomeDL v3 end-to-end verification and inference runner

python -m pytest -v

python -m genomedl.inference.vcf_pipeline data/demo/sample_demo.vcf \
    --out_report reports/demo_vcf/vcf_report.txt \
    --out_json reports/demo_vcf/vcf_report.json \
    --out_pdf reports/demo_vcf/vcf_report.pdf

# Optional: run full training & 5-fold out-of-gene evaluation
# python -m genomedl.training.train_fusion --config configs/fusion_v3.yaml --device cuda
# python -m genomedl.evaluation.evaluate --device cuda
