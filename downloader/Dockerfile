FROM python:3.12-slim

WORKDIR /app
COPY downloader/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY downloader/ .

RUN useradd --create-home --shell /usr/sbin/nologin app \
    && mkdir -p /data/images \
    && chown -R app:app /app /data

USER app

CMD ["python", "-m", "downloader"]
