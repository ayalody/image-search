FROM python:3.11-slim

# ─── Install curl for the healthcheck ───────────────────────────────
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY search_api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY common   common
COPY search_api/app app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
