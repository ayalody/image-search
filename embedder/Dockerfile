# syntax=docker/dockerfile:1

###########################
# 1) Builder stage       #
###########################
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    PIP_RETRIES=5

# system deps for PIL, torch, etc.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libglib2.0-0 libgl1 libjpeg-dev gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY embedder/requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir \
        --timeout $PIP_DEFAULT_TIMEOUT \
        --retries $PIP_RETRIES \
        --resume-retries 3 \
        -r requirements.txt

COPY common/   ./common
COPY embedder/ .

############################
# 2) Runtime stage        #
############################
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# pull in only the installed packages + your code
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /app /app

CMD ["python", "-m", "worker"]
