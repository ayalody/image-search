# 🖼️ Image‑Search Project

A fully Docker‑orchestrated demo that turns any list of image URLs into a local **vector‑search engine**.  It downloads pictures, embeds them with OpenCLIP, stores vectors in Elasticsearch 8, and serves a FastAPI + Streamlit front‑end so you can type a prompt such as *“red car at night”* and instantly see relevant images.

---

## 📊 Architecture at a glance

```mermaid
flowchart LR
    subgraph Data
        A["urls.txt"] -- "bind‑mount" --> D
    end

    subgraph Containers
        D(["Downloader<br/>(asyncio)"]) --> I[["/images volume/"]]
        E(["Embedder<br/>(OpenCLIP → vector)"]) -->|"_bulk docs_"| ES[("es<br/>(Elasticsearch 8<br/>HNSW index)")]
        I --> E
        I -. "read‑only" .-> API(["Search‑api<br/>(FastAPI)"])
        API <==>|"search"| ES
        UI(["UI<br/>(Streamlit)"]) -->|"REST /search"| API
    end

    classDef c fill:#f6f8fa,stroke:#999,color:#000;
    class D,E,ES,API,UI,I c;
```

*Shared state*

* **images volume** – raw `*.jpg/.png` files (Downloader ⇄ Embedder ⇄ UI)
* **es‑data volume** – persisted Lucene shards

---

## 🚀 Quick start (local)

```bash
# 1. clone & position at repo root
$ git clone https://github.com/ayalody/image-search.git && cd image-search

# 2. put some image URLs (one per line)
$ echo "https://picsum.photos/id/237/600/400" >> data/image_urls.txt

# 3. build & launch
$ docker compose build --pull
$ docker compose up --wait -d     # exits when every service is healthy

# 4. open UI
$ open http://localhost:8501       # or curl the API:  GET :8000/search/text
```

When the UI loads, type a phrase and you should see thumbnails in ≤ 1 second.

---

## ⚙️ Runtime configuration

| Env var (service)           | Default                | What it does                                  |
| --------------------------- | ---------------------- | --------------------------------------------- |
| \`\` (downloader)           | `/data/image_urls.txt` | Path to newline‑separated list of image URLs. |
| \`\` (downloader)           | `/data/images`         | Where JPEGs/PNGs are written.                 |
| \`\` (downloader)           | `32`                   | Socket limit for parallel downloads.          |
| \`\` (downloader, embedder) | `30`                   | How often each service re‑scans for new work. |
| \`\` (embedder)             | `/data/images`         | Directory to walk for picture files.          |
| \`\` (embedder)             | `RN50`                 | Any OpenCLIP ckpt, e.g. `ViT‑L‑14‑quickgelu`. |
| \`\` (all services)         | `http://es:9200`       | Elasticsearch URL.                            |
| \`\` (embedder)             | `INFO`                 | `DEBUG` prints idle heart‑beats.              |

Set these with `-e` flags or a `.env` file.

---

## 📈 Scaling strategy (millions of images, high traffic)

| Layer            | How to scale                                                                                                                                                                  |
| ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Downloader**   | Push URLs into SQS/RabbitMQ; autoscale multiple downloader pods; use S3 instead of local volume.                                                                              |
| **Embedder**     | Horizontal GPU workers behind queue; switch to batch‑embedding & helper.bulk; store vectors in nightly bulk jobs.                                                             |
| **Vector store** | Elastic Search → 3‑node dedicated cluster• *data* hot tier for vectors• 1 replica for HA• `m=32`, larger heap.  For >50 M images consider OpenSearch K‑NN or Faiss + DiskANN. |
|                  |                                                                                                                                                                               |
| **API**          | Gunicorn/Uvicorn with 2× vCPU workers; behind nginx ingress; enable ES HTTP compression; cache last 1 k queries in Redis.                                                     |
| **UI**           | Stateless—scale via additional Streamlit pods or migrate to React+Next.js for CDN hosting.                                                                                    |

> Total ingestion throughput becomes a function of embed‑GPU count; query throughput scales independently by adding API pods.

---

## 🧪 Development tips

```bash
# follow logs of one service
$ docker compose logs -f embedder

# run unit tests & lint (requires python 3.11 locally)
$ hatch run all        # or  tox -e py311

# wipe everything
$ docker compose down -v && docker builder prune -af
```

---

## 📝 Contributing

1. Fork / feature‑branch.
2. Pre‑commit hooks (`ruff`, `black`, `isort`).
3. `docker compose build --pull --no-cache` must stay green.
4. Submit PR; CI runs cold‑boot health check and `pip check`.

PRs for new models or Faiss back‑end are welcome!

---

© 2025 Image Search Project — MIT licensed.
