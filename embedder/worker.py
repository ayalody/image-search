"""
Walk the shared images volume, create embeddings, and push them into
Elasticsearch with an id = SHA-256(image bytes) so duplicates are skipped.
Run continuously (tail-mode) or exit after one pass if BATCH_ONCE=1.
"""

import hashlib, logging, time, os
from pathlib import Path

import torch
import PIL.Image as Image
from tqdm import tqdm

from config import settings
from common.models import load_model
from common.es_utils import ensure_index_exists, get_es_client


# ---------------------------------------------------------------------------
logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s  %(levelname)-8s %(message)s",
)

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = load_model(settings.model_name, device=device)
model.eval()           # no gradients

es = get_es_client()
ensure_index_exists(es, index="images")          # creates dense_vector mapping

def sha256_bytes(fp):
    h = hashlib.sha256()
    while chunk := fp.read(8192):
        h.update(chunk)
    fp.seek(0)
    return h.hexdigest()

def embed_single(img_path: Path):
    with img_path.open("rb") as f:
        digest = sha256_bytes(f)
        if es.exists(index="images", id=digest):
            return  # already indexed

        img = Image.open(f).convert("RGB")
        tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            vec = model.encode_image(tensor).cpu().numpy()[0]

        doc = {"path": str(img_path), "vector": vec.tolist()}
        es.index(index="images", id=digest, document=doc)

def main():
    images = list(Path(settings.images_dir).rglob("*.[jp][pn]g"))
    logging.info("Found %d images to scan", len(images))

    for img_path in tqdm(images, desc="Embedding"):
        try:
            embed_single(img_path)
        except Exception as e:
            logging.exception("Failed on %s: %s", img_path, e)

    logging.info("Done – indexed all new images")

if __name__ == "__main__":
    main()
