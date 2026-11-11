FROM python:3.11-slim

# System libraries needed for RDKit/Matplotlib headless rendering
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user with UID 1000 as required by Hugging Face Spaces
RUN useradd -m -u 1000 user

USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1

WORKDIR /home/user/app

# Install Python dependencies first for clean Docker layer caching.
# requirements.txt configures the CPU-only PyTorch wheel automatically on Linux.
COPY --chown=user:user requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip "setuptools>=68.0" && \
    pip install --no-cache-dir -r requirements.txt

# Pre-seed empty Streamlit credentials so container boots cleanly without an email prompt
RUN mkdir -p /home/user/.streamlit && \
    printf '[general]\nemail = ""\n' > /home/user/.streamlit/credentials.toml

# Copy runtime application, model weights, reference data, and package source
COPY --chown=user:user . .
RUN pip install --no-cache-dir --no-build-isolation --no-deps -e .

EXPOSE 7860

# Disable XSRF/CORS checks for Hugging Face Spaces cross-origin iframe file uploads
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0", "--server.enableXsrfProtection=false", "--server.enableCORS=false"]
