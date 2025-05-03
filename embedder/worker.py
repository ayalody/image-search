"""
Walk the shared images volume, create embeddings, and push them into
Elasticsearch with an id = SHA‑256(image bytes) so duplicates are skipped.
Run once, then quit (or loop if you later add tail‑mode logic).
"""
import hashlib, logging
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from config import settings
from common.models import encoder                    # <‑‑ singleton
from common.es_utils import ensure_index_exists, get_es_client, ES_INDEX

# ─────────────────────────────────────────── logging
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)

# ─────────────────────────────────────────── model + ES
device = "cuda" if torch.cuda.is_available() else "cpu"
# move the shared encoder to the right device once
encoder.model.to(device)

es = get_es_client()
ensure_index_exists(es, ES_INDEX)                    # guarantees correct dims

# ─────────────────────────────────────────── helpers
def sha256_bytes(fp):
    h = hashlib.sha256()
    while chunk := fp.read(8192):
        h.update(chunk)
    fp.seek(0)
    return h.hexdigest()

def embed_single(img_path: Path):
    with img_path.open("rb") as f:
        digest = sha256_bytes(f)
        if es.exists(index=ES_INDEX, id=digest):
            return  # already indexed

        img = Image.open(f).convert("RGB")
        vec = encoder.image(img)                    # returns list[float] of embed_dim
        doc = {"path": str(img_path), "vector": vec}
        es.index(index=ES_INDEX, id=digest, document=doc)

# ─────────────────────────────────────────── main loop
def main():
    images = list(Path(settings.images_dir).rglob("*.[jp][pn]g"))
    logging.info("Found %d images", len(images))

    for img_path in tqdm(images, desc="Embedding"):
        try:
            embed_single(img_path)
        except Exception as exc:
            logging.exception("Failed on %s → %s", img_path, exc)

    logging.info("Done – indexed all new images")

if __name__ == "__main__":
    main()
