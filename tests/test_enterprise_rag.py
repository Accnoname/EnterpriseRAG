"""
Unit and Integration Tests for EnterpriseRAG:
- Test 1: Data Contracts validation (Pydantic schemas)
- Test 2: DuckDB Analytical Engine and Parquet storage
- Test 3: Safe Python Sandbox (Security disallowance & accuracy)
- Test 4: Chroma Adapter & Hybrid RAG (BM25 + Dense RRF)
- Test 5: Semantic Cache Hit & Miss
"""

import pytest
from src.contracts.schemas import DocumentChunk, FinancialMetric, FinancialTeamState
from src.adapters.duckdb_adapter import DuckDBAdapter
from src.adapters.chroma_adapter import ChromaAdapter
from src.core.hybrid_retriever import HybridRetriever
from src.sandbox.runner import SafePythonSandbox
from src.cache.semantic_cache import SemanticCache


def test_data_contracts():
    chunk = DocumentChunk(
        chunk_id="TEST_01",
        entity="NVDA",
        period="2024",
        section="Item 1A",
        text="Test risk statement about TSMC."
    )
    assert chunk.chunk_id == "TEST_01"
    assert chunk.entity == "NVDA"

    metric = FinancialMetric(
        entity="MSFT",
        year=2024,
        revenue_billion=245.12,
        net_income_billion=88.14,
        operating_margin_pct=44.6
    )
    assert metric.revenue_billion == 245.12


def test_duckdb_adapter(tmp_path):
    parquet_file = tmp_path / "test_metrics.parquet"
    adapter = DuckDBAdapter(parquet_path=parquet_file)
    metrics = [
        FinancialMetric(entity="AAPL", year=2023, revenue_billion=383.29, net_income_billion=96.99, operating_margin_pct=29.8),
        FinancialMetric(entity="AAPL", year=2024, revenue_billion=391.04, net_income_billion=93.74, operating_margin_pct=31.2)
    ]
    count = adapter.load_metrics(metrics)
    assert count == 2
    res = adapter.execute_sql("SELECT entity, MAX(revenue_billion) as max_rev FROM financial_metrics GROUP BY entity")
    assert len(res) == 1
    assert round(res[0]["max_rev"], 2) == 391.04


def test_safe_python_sandbox():
    sandbox = SafePythonSandbox()
    # Test safe calculation
    res = sandbox.execute("val = round(num1 * 1.5, 2)", {"num1": 10.0})
    assert res["val"] == 15.0

    # Test security violation blocking
    with pytest.raises(ValueError):
        sandbox.execute("import sys; sys.exit(0)", {})

    with pytest.raises(ValueError):
        sandbox.execute("exec('print(1)')", {})


def test_hybrid_retriever(tmp_path):
    chroma_dir = tmp_path / "test_chroma"
    chroma = ChromaAdapter(collection_name="unit_test_col", persist_dir=chroma_dir)
    chunks = [
        DocumentChunk(chunk_id="C1", entity="NVDA", period="2024", section="Item 1A", text="High reliance on TSMC foundries."),
        DocumentChunk(chunk_id="C2", entity="MSFT", period="2024", section="Item 1A", text="Azure cloud infrastructure rivalry with AWS.")
    ]
    chroma.index_chunks(chunks)
    retriever = HybridRetriever(vector_store=chroma, chunks=chunks)

    results = retriever.search_hybrid("TSMC foundries semiconductor", top_k=1)
    assert len(results) > 0
    assert results[0].chunk.entity == "NVDA"
    assert "ID:C1" in results[0].citation


def test_semantic_cache():
    cache = SemanticCache(similarity_threshold=0.85)
    cache.put("What are the key supply chain risks for NVDA and TSMC?", "Cached response")
    hit = cache.get("What are the primary supply chain risks for NVDA with TSMC?")
    assert hit == "Cached response"
    miss = cache.get("Tell me about a completely unrelated recipe for apple pie.")
    assert miss is None

