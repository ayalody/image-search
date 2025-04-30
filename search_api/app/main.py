from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from pathlib import Path
from .schemas import TextQuery
from .config  import ES_HOST, ES_INDEX, TOP_K_DEFAULT, NUM_CANDIDATES
from common.models import encoder

from common.es_utils import get_es_client, knn_search

app = FastAPI(
    title       = "Image-Search-API",
    version     = "1.0.0",
    description = "k-NN search over image embeddings stored in Elasticsearch",
)

es = get_es_client()                 # one client per worker


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

@app.get("/healthz")
async def healthz():
    return {"status": "ok"}
