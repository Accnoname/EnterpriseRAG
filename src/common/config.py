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

# Primary & Fallback Models (Handling 503 Overload)
PRIMARY_LLM_MODEL = "gemini-3.8-flash"
FALLBACK_LLM_MODELS = ["gemini-3.5-flash", "gemini-3.5-flash-lite", "gemini-flash-latest"]
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RRF_K = 60
CACHE_SIMILARITY_THRESHOLD = 0.85
