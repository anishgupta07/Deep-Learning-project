# Deploying GenomeDL to Hugging Face Spaces

Hugging Face Spaces deprecated the native `sdk: streamlit` runtime for new Spaces, so Streamlit apps are deployed using a **Docker Space** (`sdk: docker`) listening on port `7860`.

This repository already includes:
- `Dockerfile` (non-root UID `1000`, CPU PyTorch + dependencies, editable `genomedl` install, exposes `7860`)
- `.dockerignore` (keeps runtime model weights, `data/reference/`, `data/cache/esm_cache.parquet`, `data/demo/sample_demo.vcf`, and `reports/benchmarks/eval.md` while excluding raw training files, archives, and tests)
- `.streamlit/config.toml` (headless server and clinical light theme)

## 1. Test Locally First

Run the Streamlit app locally from the repository root:

```bash
python -m streamlit run app.py
```

Wait until you see the local URL (usually `http://localhost:8501`), test both the **Demo VCF** and **Upload Custom VCF** flows, and verify PDF/JSON/TXT downloads.

*(Optional local Docker build check if Docker Desktop is installed):*

```bash
docker build -t genomedl-app .
docker run --rm -p 7860:7860 genomedl-app
```

## 2. Create the Hugging Face Space

1. Sign in to [Hugging Face](https://huggingface.co/) and click **New Space**.
2. Choose **Docker** as the Space SDK (select the **Blank** template).
3. Hardware: **CPU basic (2 vCPU · 16 GB)** is sufficient — inference runs on CPU in a couple of seconds using the precomputed ESM-2 cache (`data/cache/esm_cache.parquet`) and bundled ensemble checkpoint (`models/fusion_v3/model.pth`).
4. Note your Space repo ID, e.g. `<your-username>/genomedl-clinical-hereditary`.

> [!NOTE]
> The default `Dockerfile` installs the CPU-only wheel for `torch>=2.4.0` so CPU Spaces build quickly without downloading ~3 GB of CUDA libraries. If you deploy onto a GPU Space later, remove that `torch` `--index-url` line in `Dockerfile`.

## 3. Configure the Space `README.md` Header

Hugging Face reads Space configuration from YAML frontmatter at the top of the Space repository's `README.md`. When pushing to the Space remote (or via `huggingface_hub`), prepend the following YAML header to the Space `README.md`:

```yaml
---
title: GenomeDL — Hereditary Variant Interpretation
emoji: 🧬
colorFrom: blue
colorTo: slate
sdk: docker
app_port: 7860
pinned: false
license: mit
---
```

Keep this YAML frontmatter in the Space's `README.md` (or on your HF deployment branch) so your main GitHub `README.md` stays clean.

## 4. Upload Using `huggingface_hub` (Recommended)

Because `models/fusion_v3/model.pth` and `data/cache/esm_cache.parquet` are large binary files (~25–150 MB+), using `huggingface_hub`'s `upload_folder` avoids Git LFS setup headaches:

```bash
pip install -U huggingface_hub
hf auth login
```

Create or update the Space `README.md` and upload the runtime workspace cleanly:

```python
from pathlib import Path
from huggingface_hub import HfApi

SPACE_ID = "<your-username>/genomedl-clinical-hereditary"

api = HfApi()
api.create_repo(
    repo_id=SPACE_ID,
    repo_type="space",
    space_sdk="docker",
    space_hardware="cpu-basic",
    exist_ok=True,
)

# Upload Space README.md with required Docker YAML frontmatter first
hf_header = """---
title: GenomeDL — Hereditary Variant Interpretation
emoji: 🧬
colorFrom: blue
colorTo: slate
sdk: docker
app_port: 7860
pinned: false
license: mit
---

"""
readme_body = Path("README.md").read_text(encoding="utf-8")
api.upload_file(
    path_or_fileobj=(hf_header + readme_body).encode("utf-8"),
    path_in_repo="README.md",
    repo_id=SPACE_ID,
    repo_type="space",
    commit_message="Configure Space Docker YAML metadata",
)

# Upload runtime repository respecting .dockerignore exclusions (keep YAML README intact)
api.upload_folder(
    folder_path=".",
    repo_id=SPACE_ID,
    repo_type="space",
    commit_message="Upload GenomeDL runtime app and model",
    ignore_patterns=[
        "README.md",
        ".git/**",
        ".venv/**",
        "**/__pycache__/**",
        ".pytest_cache/**",
        "archive/**",
        "docs/**",
        "tests/**",
        "scripts/**",
        "data/raw/**",
        "data/processed/v3_featured_snv.csv",
        "data/cache/esm_embeddings.pt",
        "reports/demo_vcf/**",
        "reports/giab_hg002/**",
    ],
)
```

## 5. Monitor the Build

Once uploaded, open `https://huggingface.co/spaces/<your-username>/genomedl-clinical-hereditary` and watch the **Logs** tab. After the Docker build completes, the Space transitions to **Running** and serves the Streamlit app at port `7860`.
