"""
ChromaDB Vector Adapter: Implements BaseVectorStore using local ChromaDB and SentenceTransformer.
"""

from typing import List, Optional, Dict, Any
from pathlib import Path
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from src.interfaces.ports import BaseVectorStore
from src.contracts.schemas import DocumentChunk, RetrievalCandidate
from src.common.config import VECTOR_DB_DIR, DEFAULT_EMBEDDING_MODEL
from src.common.logger import get_logger

logger = get_logger("ChromaAdapter")


class ChromaAdapter(BaseVectorStore):
    def __init__(self, collection_name: str = "enterprise_knowledge", persist_dir: Path = VECTOR_DB_DIR):
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir), settings=Settings(anonymized_telemetry=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.model = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        if not chunks:
            return 0
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [{
            "entity": c.entity,
            "period": c.period,
            "section": c.section,
            "source": c.source
        } for c in chunks]

        embeddings = self.model.encode(texts, convert_to_numpy=True).tolist()
        self.collection.upsert(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
        logger.info(f"Indexed {len(chunks)} chunks into Chroma collection '{self.collection.name}'")
        return len(chunks)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_entity: Optional[str] = None
    ) -> List[RetrievalCandidate]:
        query_emb = self.model.encode([query], convert_to_numpy=True).tolist()
        where_clause = {"entity": filter_entity} if filter_entity else None

        results = self.collection.query(
            query_embeddings=query_emb,
            n_results=top_k,
            where=where_clause
        )

        candidates = []
        if results and results["ids"] and results["ids"][0]:
            for rank, (cid, doc, meta, dist) in enumerate(
                zip(results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]),
                start=1
            ):
                similarity = 1.0 - dist  # Cosine distance to similarity
                chunk = DocumentChunk(
                    chunk_id=cid,
                    entity=meta.get("entity", "UNKNOWN"),
                    period=meta.get("period", "UNKNOWN"),
                    section=meta.get("section", "UNKNOWN"),
                    text=doc,
                    source=meta.get("source", "UNKNOWN"),
                    metadata=meta
                )
                citation = f"[{chunk.entity} | {chunk.period} | {chunk.section} | ID:{chunk.chunk_id}]"
                candidates.append(RetrievalCandidate(
                    chunk=chunk,
                    score=float(similarity),
                    rank=rank,
                    citation=citation
                ))
        return candidates

