"""
Utilities shared functions by downloader / embedder / search-api.
"""
import os
import logging
from elasticsearch import Elasticsearch, ConnectionError, TransportError, BadRequestError

# ────────────────────────────────────────────────────────────────────────────────
# CONFIG
ES_HOST   = os.getenv("ES_HOST", "http://es:9200")   # overridable in compose
ES_INDEX  = os.getenv("ES_INDEX", "images")
TIMEOUT_S = int(os.getenv("ES_TIMEOUT", 30))
VECTOR_DIM = 768        # 768 for ViT-L/14; change when loading another checkpoint

_MAPPING = {
    "mappings": {
        "properties": {
            "path":   {"type": "keyword"},
            "vector": {
                "type": "dense_vector",
                "dims": VECTOR_DIM,
                "index": True,
                "similarity": "cosine",
                "index_options": {"type": "hnsw", "m": 16, "ef_construction": 512},
            },
        }
    }
}

def ensure_index_exists(es: Elasticsearch, index: str = "images"):
    """
    Create the 'images' index with an ANN-enabled dense_vector field.
    Safe to call multiple times (no-op if already there).
    """
    try:
        if es.indices.exists(index=index):
            return
    except BadRequestError:            # corrupt stub → drop & recreate
        es.indices.delete(index=index, ignore_unavailable=True)

    es.indices.create(index=index, body=_MAPPING)


# ────────────────────────────────────────────────────────────────────────────────
# CLIENT FACTORY
def get_es_client() -> Elasticsearch:
    """
    Return a single Elasticsearch client instance.
    Call this once per-process (safe to reuse between requests).
    """
    es = Elasticsearch(
        hosts=[ES_HOST],
        request_timeout=TIMEOUT_S,
        max_retries=3,
        retry_on_timeout=True,
    )
    try:
        es.info()                     # ping
    except (ConnectionError, TransportError) as exc:
        logging.error("Cannot reach Elasticsearch at %s → %s", ES_HOST, exc)
        raise
    return es


# ────────────────────────────────────────────────────────────────────────────────
# HELPER FOR K-NN QUERIES
def knn_search(
    es: Elasticsearch,
    index: str,
    vector: list[float],
    k: int = 10,
    candidates: int = 100,
    source_fields: list[str] | None = None,
):
    """
    Convenience wrapper around es.knn_search().
    Returns a list of {id, score, ..._source} dicts.
    """
    source_fields = source_fields or ["path"]
    resp = es.knn_search(
        index=index,
        knn={
            "field": "vector",
            "query_vector": vector,
            "k": k,
            "num_candidates": candidates,
        },
        _source=source_fields,
    )
    return [
        {**hit["_source"], "id": hit["_id"], "score": hit["_score"]}
        for hit in resp["hits"]["hits"]
    ]

