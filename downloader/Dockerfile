FROM python:3.12-slim

WORKDIR /app
COPY downloader/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY downloader/ .

CMD ["python", "downloader.py"]
