"""
Domain Data Contracts: Pydantic schemas enforcing strict data validation
across Ingestion, Vector Stores, OLAP Storage, and Multi-Agent communications.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    """Normalized unstructured text chunk representing a semantic unit of text."""
    chunk_id: str = Field(description="Unique identifier (e.g. hash or TICKER_YEAR_INDEX)")
    entity: str = Field(description="Entity or Ticker name, e.g. NVDA, AAPL")
    period: str = Field(description="Time period or fiscal year, e.g. 2024")
    section: str = Field(description="Section heading or topic name, e.g. Item 1A - Risk Factors")
    text: str = Field(description="The primary textual content")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary metadata attributes")
    source: str = Field(default="SEC_10K", description="Data origin identifier")


class FinancialMetric(BaseModel):
    """Normalized quantitative metric record suitable for analytical queries."""
    entity: str = Field(description="Entity or Ticker symbol, e.g. NVDA, MSFT")
    year: int = Field(description="Fiscal reporting year, e.g. 2024")
    revenue_billion: float = Field(description="Total revenue in Billions USD")
    net_income_billion: float = Field(description="Net income in Billions USD")
    operating_margin_pct: float = Field(description="Operating margin as a percentage")
    fcf_billion: Optional[float] = Field(default=None, description="Free cash flow in Billions USD")
    source: str = Field(default="SEC_XBRL_FACTS", description="Source document type")


class RetrievalCandidate(BaseModel):
    """Output candidate from Hybrid Retrieval with rank scores and citations."""
    chunk: DocumentChunk
    score: float = Field(description="Combined relevance score (e.g. RRF score)")
    rank: int = Field(description="Candidate rank in retrieved list")
    citation: str = Field(description="Legal/Document citation reference string")


class FinancialTeamState(BaseModel):
    """Shared state for the Stateful Multi-Agent Supervisor workflow."""
    user_query: str = Field(description="Original user request")
    target_entities: List[str] = Field(default_factory=list, description="Target companies/tickers")
    qualitative_risks: Optional[Dict[str, Any]] = Field(default=None, description="Findings from Risk Analyst Worker")
    quantitative_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Metrics calculated from OLAP/Python")
    audit_trail: List[str] = Field(default_factory=list, description="Chronological log of agent actions")
    iteration_count: int = Field(default=0, description="Guard against infinite recursion")
    final_memo: Optional[str] = Field(default=None, description="Executive synthesized investment memorandum")

