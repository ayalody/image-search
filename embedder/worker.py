"""
Embedder – poll mode
Re‑walk the shared images volume every POLL_SECONDS seconds, create
embeddings for files that are not yet in Elasticsearch, then sleep.
"""

import hashlib, logging, os, time
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image
from tqdm import tqdm

from config import settings
from common.models import encoder                      # singleton
from common.es_utils import ensure_index_exists, get_es_client, ES_INDEX

# ── tunables ──────────────────────────────────────────────────────────────
POLL_SECONDS = int(os.getenv("POLL_SECONDS", 30))
IMAGES_DIR   = Path(settings.images_dir)               # usually /data/images
# ──────────────────────────────────────────────────────────────────────────

# ── logging ---------------------------------------------------------------
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)

# ── model + Elasticsearch -------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
encoder.model.to(device)

es = get_es_client()
ensure_index_exists(es, ES_INDEX)                      # guarantees correct dims

# ── helpers ---------------------------------------------------------------
def sha256_bytes(fp) -> str:
    h = hashlib.sha256()
    while chunk := fp.read(8192):
        h.update(chunk)
    fp.seek(0)
    return h.hexdigest()

def iter_images() -> Iterable[Path]:
    return IMAGES_DIR.rglob("*.[jp][pn]g")

def embed_and_index(img_path: Path) -> bool:
    """Return True if a new document was written, False otherwise."""
    with img_path.open("rb") as f:
        digest = sha256_bytes(f)
        if es.exists(index=ES_INDEX, id=digest):
            return False                            # already indexed

        img = Image.open(f).convert("RGB")
        vec = encoder.image(img)                    # returns list[float] of embed_dim
        doc = {"path": str(img_path), "vector": vec}
        es.index(index=ES_INDEX, id=digest, document=doc)
        return True

# ── main loop -------------------------------------------------------------
def run_once() -> int:
    """Embed any not‑yet‑indexed images. Return number processed."""
    images = list(iter_images())
    new_count = 0

    for img_path in tqdm(images, desc="Embedding", leave=False):
        try:
            if embed_and_index(img_path):
                new_count += 1
        except Exception as exc:
            logging.exception("Failed on %s → %s", img_path, exc)

    return new_count

def main_loop():
    logging.info("Embedder started – polling every %s s", POLL_SECONDS)
    while True:
        added = run_once()
        if added:
            logging.info("✓ round complete – %d new embedding(s)", added)
        time.sleep(POLL_SECONDS)

if __name__ == "__main__":
    try:
        main_loop()
    except KeyboardInterrupt:
        logging.info("Shutting down")
