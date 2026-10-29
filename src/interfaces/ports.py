"""
Abstract Interfaces (Ports) for the Hexagonal Architecture.
Any Vector Database or Analytical Engine can plug into these contracts.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from src.contracts.schemas import DocumentChunk, RetrievalCandidate, FinancialMetric


class BaseVectorStore(ABC):
    """Port for any Vector DB (ChromaDB, Qdrant, Milvus, pgvector)."""

    @abstractmethod
    def index_chunks(self, chunks: List[DocumentChunk]) -> int:
        """Indexes a list of DocumentChunks. Returns count of indexed documents."""
        pass

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_entity: Optional[str] = None
    ) -> List[RetrievalCandidate]:
        """Performs vector similarity search."""
        pass


class BaseQueryEngine(ABC):
    """Port for any Structured Analytical Query Engine (DuckDB, Snowflake, BigQuery)."""

    @abstractmethod
    def execute_sql(self, sql_query: str) -> List[Dict[str, Any]]:
        """Executes a SQL query and returns rows as dictionaries."""
        pass

    @abstractmethod
    def load_metrics(self, metrics: List[FinancialMetric]) -> int:
        """Loads structured financial records into the table store."""
        pass


class BaseAgentWorker(ABC):
    """Port for any specialized Worker Agent."""

    @abstractmethod
    def process_task(self, task_instruction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Executes an assigned task and returns structured findings."""
        pass

