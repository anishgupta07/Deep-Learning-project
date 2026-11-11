# Deploying GenomeDL to Streamlit Community Cloud

Streamlit Community Cloud hosts `app.py` directly from GitHub for free (no credit card required, up to ~2.7 GB dynamic RAM, ideal for our ~889 MB CPU runtime footprint).

## Repository Deployment Files

- `app.py` — Streamlit UI entrypoint (automatically adds `src/` to `sys.path` so no editable install step is needed on Streamlit Cloud).
- `requirements.txt` — Installs `torch==2.6.0+cpu` from PyTorch's CPU wheel index on Linux (`--extra-index-url https://download.pytorch.org/whl/cpu`) to keep cloud builds fast and lightweight without ~3 GB of CUDA libraries.
- `packages.txt` — Installs Debian runtime libraries (`libglib2.0-0`, `libxext6`, `libxrender1`) needed for RDKit headless rendering.
- `.streamlit/config.toml` — Clinical light theme, `headless = true`, `maxUploadSize = 200`, and usage stats disabled.

## 1. Push Changes to GitHub

Ensure all refactored code, `app.py`, `requirements.txt`, `packages.txt`, `.streamlit/config.toml`, and model/reference assets are committed and pushed to your GitHub branch (for example `main`):

```bash
git add -A
git commit -m "Refactor genomedl package and add Streamlit frontend"
git push origin main
```

All required runtime files (`models/fusion_v3/model.pth` ~6.7 MB, `data/cache/esm_cache.parquet` ~7.2 MB, `data/processed/clinvar_labelled_v3.csv` ~28.8 MB) are under GitHub's 100 MB file limit, while larger files (`data/raw/`, `data/cache/esm_embeddings.pt`) are excluded by `.gitignore`.

## 2. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with **Continue with GitHub**.
2. Click **Create app** (or **New app**) → **Deploy a public app from GitHub** (or **Yup, I have an app**).
3. Select:
   - **Repository:** `anishgupta07/Deep-Learning-project`
   - **Branch:** the branch you pushed (`main` or `ftp-dataset`)
   - **Main file path:** `app.py`
4. Open **Advanced settings...** and select **Python 3.11** (or **3.12**).
5. Click **Deploy!**

## 3. Verify Live Startup

- Initial package installation + first ClinVar index build takes ~1–3 minutes (`@st.cache_resource` keeps the pipeline cached in memory afterward).
- Once live, confirm the synthetic Demo VCF auto-loads `5` variants (`2` actionable, `1` carrier), test uploading a `.vcf` / `.vcf.gz`, and download the PDF/JSON/TXT clinical reports.
