from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

import os
from pathlib import Path

# local deps
from .schemas import TextQuery
from .config  import ES_INDEX, TOP_K_DEFAULT, NUM_CANDIDATES, DEVICE

# shared utils
from common.models   import encoder
from common.es_utils import get_es_client, ensure_index_exists, knn_search


app = FastAPI(
    title       = "Image-Search-API",
    version     = "1.0.0",
    description = "k-NN search over image embeddings stored in Elasticsearch",
)

# -------------------------- ES client + index -------------------------------------
es = get_es_client()                 # one client per process
ensure_index_exists(es, ES_INDEX)    # create / fix mapping on startup

# -------------------------- ensure encoder on right device ------------------------
encoder.model.to(DEVICE)


# ─────────────────────────── ROUTES ──────────────────────────
@app.post("/search/text")
async def search_text(body: TextQuery):
    """
    Encode user prompt → CLIP vector → k-NN in Elasticsearch.
    """
    k   = body.k or TOP_K_DEFAULT
    vec = encoder.text(body.text)

    hits = knn_search(
        es, index=ES_INDEX, vector=vec,
        k=k, candidates=NUM_CANDIDATES,
        source_fields=["path"]        # adjust if you index more meta-data
    )
    for h in hits:
        filename = Path(h["path"]).name          # strip /data/images/…
        h["url"]  = f"/images/{filename}"        # add public URL
        del h["path"]                            # optional: hide internals

    return JSONResponse(hits)


@app.post("/search/image")
async def search_image(file: UploadFile, k: int = TOP_K_DEFAULT):
    """
    Upload an image (any common type). Returns k most similar images.
    """
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(400, "Uploaded file must be an image")

    img_bytes = await file.read()
    vec       = encoder.image(img_bytes)

    hits = knn_search(
        es, index=ES_INDEX, vector=vec,
        k=k, candidates=NUM_CANDIDATES,
        source_fields=["path"]
    )
    for h in hits:
        filename = Path(h["path"]).name          # strip /data/images/…
        h["url"]  = f"/images/{filename}"        # add public URL
        del h["path"]                            # optional: hide internals

    return JSONResponse(hits)

app.mount(
    "/images",
    StaticFiles(directory="/data/images"),
    name="images",
)

@app.get("/meta")
async def meta() -> dict:
    info   = es.info()
    health = es.cluster.health()
    count  = es.count(index=ES_INDEX)["count"]

    return {
        "model_name":  os.getenv("MODEL", "RN50"),
        "vector_dim":  encoder.embed_dim,
        "device":      str(next(iter(encoder.model.parameters())).device),
        "es_version":  info["version"]["number"],
        "es_index":    ES_INDEX,
        "doc_count":   count,
        "hnsw_m":      16,
        "hnsw_ef":     512,
        "cluster":     health["status"],
    }


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
