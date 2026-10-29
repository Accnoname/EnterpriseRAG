"""
Hybrid Retriever combining BM25 Lexical Keyword search and Dense Vector embeddings via RRF.
"""

from typing import List, Optional, Dict
from rank_bm25 import BM25Okapi
from src.interfaces.ports import BaseVectorStore
from src.contracts.schemas import DocumentChunk, RetrievalCandidate
from src.common.config import RRF_K
from src.common.logger import get_logger

logger = get_logger("HybridRetriever")


class HybridRetriever:
    def __init__(self, vector_store: BaseVectorStore, chunks: Optional[List[DocumentChunk]] = None):
        self.vector_store = vector_store
        self.chunks_map: Dict[str, DocumentChunk] = {}
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_ids: List[str] = []

        if chunks:
            self.init_bm25(chunks)

    def init_bm25(self, chunks: List[DocumentChunk]):
        self.chunks_map = {c.chunk_id: c for c in chunks}
        self.chunk_ids = [c.chunk_id for c in chunks]
        corpus_tokens = [c.text.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(corpus_tokens)
        logger.info(f"Initialized BM25 index over {len(chunks)} documents")

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        filter_entity: Optional[str] = None,
        weight_dense: float = 1.0,
        weight_bm25: float = 1.0
    ) -> List[RetrievalCandidate]:
        """Executes Reciprocal Rank Fusion (RRF) between Dense Vector & BM25."""
        # 1. Dense Vector Search
        dense_results = self.vector_store.search(query, top_k=top_k * 2, filter_entity=filter_entity)

        # 2. Sparse BM25 Search
        bm25_ranked_ids = []
        if self.bm25 and self.chunk_ids:
            query_tokens = query.lower().split()
            doc_scores = self.bm25.get_scores(query_tokens)
            sorted_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)
            for idx in sorted_indices[:top_k * 2]:
                cid = self.chunk_ids[idx]
                chunk = self.chunks_map[cid]
                if filter_entity and chunk.entity != filter_entity:
                    continue
                bm25_ranked_ids.append(cid)

        # 3. Reciprocal Rank Fusion
        rrf_scores: Dict[str, float] = {}
        candidate_docs: Dict[str, DocumentChunk] = {}

        # Dense RRF scoring
        for rank, cand in enumerate(dense_results, start=1):
            cid = cand.chunk.chunk_id
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (weight_dense / (RRF_K + rank))
            candidate_docs[cid] = cand.chunk

        # BM25 RRF scoring
        for rank, cid in enumerate(bm25_ranked_ids, start=1):
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (weight_bm25 / (RRF_K + rank))
            if cid not in candidate_docs and cid in self.chunks_map:
                candidate_docs[cid] = self.chunks_map[cid]

        # 4. Sort and return top candidates
        sorted_cids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        fused_candidates = []
        for rank, cid in enumerate(sorted_cids, start=1):
            chunk = candidate_docs[cid]
            citation = f"[{chunk.entity} | {chunk.period} | {chunk.section} | ID:{chunk.chunk_id}]"
            fused_candidates.append(RetrievalCandidate(
                chunk=chunk,
                score=float(rrf_scores[cid]),
                rank=rank,
                citation=citation
            ))

        return fused_candidates

