"""
Semantic Cache Layer: Caches user queries and answers using vector cosine similarity.
Reduces recurring query latency to < 10ms and token costs to $0.
"""

from typing import Optional, Dict
import numpy as np
from sentence_transformers import SentenceTransformer
from src.common.config import DEFAULT_EMBEDDING_MODEL, CACHE_SIMILARITY_THRESHOLD
from src.common.logger import get_logger

logger = get_logger("SemanticCache")


class SemanticCache:
    def __init__(self, similarity_threshold: float = CACHE_SIMILARITY_THRESHOLD):
        self.similarity_threshold = similarity_threshold
        self.model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
        self.cache_entries: list[dict] = []  # List of {query: str, embedding: np.ndarray, response: str}

    def get(self, query: str) -> Optional[str]:
        if not self.cache_entries:
            return None

        query_emb = self.model.encode(query, convert_to_numpy=True)
        query_norm = np.linalg.norm(query_emb)

        for entry in self.cache_entries:
            cached_emb = entry["embedding"]
            similarity = np.dot(query_emb, cached_emb) / (query_norm * np.linalg.norm(cached_emb))
            if similarity >= self.similarity_threshold:
                logger.info(f"⚡ [SEMANTIC CACHE HIT] Similarity: {similarity:.4f} for query: '{query}'")
                return entry["response"]

        logger.info(f"🔍 [SEMANTIC CACHE MISS] for query: '{query}'")
        return None

    def put(self, query: str, response: str):
        query_emb = self.model.encode(query, convert_to_numpy=True)
        self.cache_entries.append({
            "query": query,
            "embedding": query_emb,
            "response": response
        })
        logger.info(f"Cached semantic entry. Total cached: {len(self.cache_entries)}")

