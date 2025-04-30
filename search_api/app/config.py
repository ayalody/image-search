"""
Centralised defaults so they can be overridden via env-vars
in docker-compose.yml or the host shell.
"""
import os

ES_HOST        = os.getenv("ES_HOST",  "http://es:9200")
ES_INDEX       = os.getenv("ES_INDEX", "images")
TOP_K_DEFAULT  = int(os.getenv("TOP_K",  10))
NUM_CANDIDATES = int(os.getenv("NUM_CANDIDATES", 100))
DEVICE         = os.getenv("DEVICE", "cpu")          # cpu | cuda
