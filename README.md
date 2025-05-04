# Image Search System

A distributed system for searching and retrieving images using vector embeddings and semantic search capabilities.

## System Architecture

```text
┌────────────┐      ┌─────────────┐      ┌────────────────┐
│  Image     │      │  Feature    │      │ Elasticsearch  │
│  Folder    │ ───▶ │  Extractor  │ ───▶ │  (Vector &     │
│ (*.jpg)    │      │  (CLIP)     │      │   Metadata)    │
└────────────┘      └─────────────┘      └────────┬───────┘
                                                  │
                          ┌───────────────────────┴──────────┐
                          │  Streamlit Web UI (inside Docker)│
                          │  • Text box → vector query       │
                          │  • Top‑k similar images grid     │
                          └──────────────────────────────────┘
```

### The system consists of several microservices working together:

- **Elasticsearch (es)**: Vector store for storing and searching image embeddings
- **Downloader**: Downloads images from provided URLs
- **Embedder**: Generates vector embeddings for images using ML models
- **Search API**: REST API for querying and retrieving images
- **Web UI**: User interface for interacting with the image search system

---

## Prerequisites

- Docker and Docker Compose
- Sufficient disk space for image storage (managed by Docker volumes)
- A list of image URLs in `data/image_urls.txt`

## Getting Started

1. Clone the repository
2. Create a `data` directory and add your `image_urls.txt` file
3. Start the services:
   ```bash
   docker-compose up -d
   ```

The system will:
- Start Elasticsearch for vector storage
- Download images from the provided URLs (stored in Docker volumes)
- Generate embeddings for the images
- Make the search API available at `http://localhost:8000`
- Serve the web UI at `http://localhost:8501`

## Data Storage

The system uses Docker volumes for data persistence:

- `images`: Stores downloaded images (shared between downloader and embedder)
- `es-data`: Stores Elasticsearch data and indices

Images are stored in Docker volumes and are not written to the host machine. This provides better isolation and performance.

## Components

### Elasticsearch (es)
- Runs on port 9200
- Stores image embeddings and metadata
- Provides vector search capabilities
- Data persisted in `es-data` volume

### Downloader
- Downloads images from URLs in `image_urls.txt`
- Stores images in shared `images` volume
- Updates Elasticsearch with image metadata

### Embedder
- Generates vector embeddings for images
- Supports multiple embedding models (e.g., OpenCLIP, BLIP2)
- Updates Elasticsearch with image embeddings
- Reads images from shared `images` volume

### Search API
- REST API available at `http://localhost:8000`
- Provides endpoints for:
  - Image search
  - Health checks
  - System status
- Accesses images through Docker volumes

### Web UI
- Streamlit-based interface
- Accessible at `http://localhost:8501`
- Provides visual interface for image search

## Configuration

- Adjust Elasticsearch heap size in `docker-compose.yml` if needed
- Configure embedding model in embedder service
- Modify ports if they conflict with existing services

## Monitoring

- Elasticsearch health: `http://localhost:9200/_cluster/health`
- Search API health: `http://localhost:8000/healthz`
- Web UI: `http://localhost:8501`

## Troubleshooting

If services fail to start:
1. Check Docker volume space availability
2. Verify `image_urls.txt` format
3. Check Elasticsearch health
4. Review container logs using `docker-compose logs <service_name>`

## Data Management

To clean up data:
```bash
# Remove all containers and volumes
docker-compose down -v

# Remove specific volumes
docker volume rm image-search_images image-search_es-data
```
