import os
import sys

# Disable annoying symlink warnings and tokenizer parallelism
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Force UTF-8 on Windows stdout/stderr
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

from pathlib import Path

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BRONZE_DIR = DATA_DIR / "bronze"
SILVER_DIR = DATA_DIR / "silver"
GOLD_DIR = DATA_DIR / "gold"
CURATED_DIR = DATA_DIR / "curated"
VECTOR_DB_DIR = CURATED_DIR / "vector_db"

# Ultrafast & Stable Production Models
# Benchmark: gemini-3.1-flash-lite delivers ~1.4s responses with 0 errors
PRIMARY_LLM_MODEL = "gemini-3.1-flash-lite"
FALLBACK_LLM_MODELS = ["gemini-3.5-flash-lite", "gemini-3.6-flash"]
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60
CACHE_SIMILARITY_THRESHOLD = 0.85
