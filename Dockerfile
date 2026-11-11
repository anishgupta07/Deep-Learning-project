# ==============================================================================
# GenomeDL v3: Production Dockerfile for Hugging Face Spaces & Cloud Microservices
# ==============================================================================
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PORT=7860 \
    HOME=/home/user

# Install system dependencies (RDKit biophysics rendering, fonts, OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgomp1 \
    libgl1 \
    libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces requires non-root user UID 1000
RUN useradd -m -u 1000 user
WORKDIR /app

# Upgrade pip and install CPU-only PyTorch first (drastically reduces image size & build time)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Copy and install Python dependencies
COPY --chown=user:user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy repository assets
COPY --chown=user:user data/ /app/data/
COPY --chown=user:user models/ /app/models/
COPY --chown=user:user src/ /app/src/
COPY --chown=user:user reports/ /app/reports/
COPY --chown=user:user app.py /app/app.py

# Switch to non-root user
USER user

# Hugging Face Spaces standard port
EXPOSE 7860

# Health check to ensure Uvicorn + Pipeline readiness
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -f http://localhost:7860/healthz || exit 1

# Start FastAPI + Gradio server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
