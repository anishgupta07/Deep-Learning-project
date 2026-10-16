# GenomeDL: Legacy & Pilot Experiments Archive

This directory stores early-stage exploratory scripts, pilot models, and initial design specifications that preceded the **GenomeDL v2 Pan-Cancer 80-Gene Engine**. They are preserved here for scientific reproducibility and historical reference.

---

### Archived Assets

| File | Description | Reason for Archival |
| :--- | :--- | :--- |
| `context.md` | Legacy somatic architecture specification (12-class, 201-column bag MIL model). | Replaced by constitutional germline 80-gene pan-cancer framework. |
| `MASTER_IMPLEMENTATION_PLAN.md` | Initial high-throughput SRA pipeline plan. | SRA streaming proved unfeasible for constitutional variant calling; replaced by ClinVar ground truth. |
| `build_brca_germline_dataset.py` | Pilot data ingestion script limited to BRCA1 and BRCA2. | Scaled to the 93-gene pan-cancer pipeline (`build_pan_cancer_germline_dataset.py`). |
| `brca_germline_pilot_dataset.csv` | Pilot dataset (BRCA1/2 only, ~11k variants). | Superseded by `pan_cancer_germline_dataset.csv` (119k variants across 93 genes). |
| `germline_brca_tabular_mlp.pth` | 2-layer pilot MLP model trained on BRCA1/2. | Superseded by `pan_cancer_germline_resnet.pth` (Tabular ResNet with skip connections & entity embeddings). |
| `verify_model.py` | Unit test script for the pilot BRCA model. | Superseded by production CLI `predict.py`. |
| `import_all_359k_variants.py` | Temporary ClinVar data audit script. | One-time diagnostic script used during dataset reconciliation. |

---

### Active Production Architecture
For active model training, evaluation, and clinical inference, refer to the root directory:
* **Inference CLI:** [`predict.py`](../../predict.py)
* **Training Pipeline:** [`train_tabular_mlp.py`](../../train_tabular_mlp.py)
* **Biochemistry Engine:** [`bio_chemistry_engine.py`](../../bio_chemistry_engine.py)
* **Dataset Builder:** [`build_pan_cancer_germline_dataset.py`](../../build_pan_cancer_germline_dataset.py)
