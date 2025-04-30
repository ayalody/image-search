# ---- build stage -----------------------------------------------------------
    FROM python:3.11-slim AS base

    # avoid interactive prompts & enable faster installs
    ENV PIP_NO_CACHE_DIR=1 \
        PYTHONDONTWRITEBYTECODE=1 \
        POETRY_VERSION=1.8.2
    
    # system dependencies for PIL, torch, etc.
    RUN apt-get update && apt-get install -y --no-install-recommends \
            libglib2.0-0 libgl1 libjpeg-dev && \
        rm -rf /var/lib/apt/lists/*
    
    WORKDIR /app
    COPY embedder/requirements.txt .
    RUN pip install --upgrade pip && pip install -r requirements.txt
    
    COPY common/   ./common
    COPY embedder/ .
    
    # ---- runtime stage ---------------------------------------------------------
    CMD ["python", "-m", "worker"]
    