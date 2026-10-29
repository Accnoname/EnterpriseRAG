"""
Automated End-to-End Self-Testing Verification Suite:
Loads mock financial data, indexes into DuckDB & Chroma, executes Hybrid RAG,
runs the Python Sandbox, tests Semantic Caching, and verifies the Multi-Agent Supervisor.
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure root path
ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from src.contracts.schemas import DocumentChunk, FinancialMetric
from src.adapters.duckdb_adapter import DuckDBAdapter
from src.adapters.chroma_adapter import ChromaAdapter
from src.core.hybrid_retriever import HybridRetriever
from src.sandbox.runner import SafePythonSandbox
from src.cache.semantic_cache import SemanticCache
from src.agents.supervisor import RiskAnalystWorker, QuantitativeMetricWorker, InvestmentSupervisorAgent
from src.common.logger import get_logger

logger = get_logger("EndToEndVerification")


def run_all_checks():
    logger.info("=================================================================")
    logger.info("[START] AUTOMATED END-TO-END VERIFICATION SUITE")
    logger.info("=================================================================")

    # 1. Test Contracts & DuckDB Adapter
    logger.info("--> [1/5] Verifying DuckDB Analytical Engine & Parquet Storage...")
    duck_adapter = DuckDBAdapter()
    sample_metrics = [
        FinancialMetric(entity="NVDA", year=2023, revenue_billion=26.97, net_income_billion=4.37, operating_margin_pct=16.2),
        FinancialMetric(entity="NVDA", year=2024, revenue_billion=60.92, net_income_billion=29.76, operating_margin_pct=48.8),
        FinancialMetric(entity="MSFT", year=2023, revenue_billion=211.91, net_income_billion=72.36, operating_margin_pct=41.8),
        FinancialMetric(entity="MSFT", year=2024, revenue_billion=245.12, net_income_billion=88.14, operating_margin_pct=44.6),
    ]
    loaded_count = duck_adapter.load_metrics(sample_metrics)
    assert loaded_count == 4, "DuckDB failed to load metric rows"
    res = duck_adapter.execute_sql("SELECT entity, AVG(operating_margin_pct) as avg_margin FROM financial_metrics GROUP BY entity")
    assert len(res) == 2, "DuckDB SQL execution failed"
    logger.info(f"[PASS] DuckDB Verified! Result: {res}")

    # 2. Test Safe Python Sandbox
    logger.info("--> [2/5] Verifying Safe Python Sandbox Execution...")
    sandbox = SafePythonSandbox()
    test_calc = sandbox.execute("pct_change = round(((y2 - y1) / y1) * 100, 2)", {"y1": 26.97, "y2": 60.92})
    assert test_calc["pct_change"] == 125.88, f"Sandbox math calculation incorrect: {test_calc}"

    # Verify security disallowance
    try:
        sandbox.execute("import os; os.system('echo test')", {})
        assert False, "Sandbox failed to block dangerous import"
    except ValueError:
        logger.info("[PASS] Sandbox Security Verified (Blocked unauthorized import successfully)!")

    # 3. Test ChromaDB & Hybrid Retriever
    logger.info("--> [3/5] Verifying Chroma Vector Store & Hybrid BM25+Dense...")
    chroma_adapter = ChromaAdapter(collection_name="test_knowledge")
    sample_chunks = [
        DocumentChunk(
            chunk_id="NVDA_2024_01",
            entity="NVDA",
            period="2024",
            section="Item 1A - Risk Factors",
            text="We rely extensively on third-party foundries such as TSMC to manufacture our semiconductor wafers. Any supply disruption at TSMC or geographic conflict in Taiwan would severely impair our chip shipping capacity and GPU production."
        ),
        DocumentChunk(
            chunk_id="MSFT_2024_01",
            entity="MSFT",
            period="2024",
            section="Item 1A - Risk Factors",
            text="Our cloud infrastructure operations and Azure AI services require massive capital expenditure in hyperscale datacenters, specialized AI accelerators, and electrical power capacity. Intense competition from AWS and Google Cloud could pressure margins."
        )
    ]
    chroma_adapter.index_chunks(sample_chunks)
    retriever = HybridRetriever(vector_store=chroma_adapter, chunks=sample_chunks)
    hybrid_res = retriever.search_hybrid(query="TSMC foundry semiconductor supplier supply chain", top_k=1)
    assert len(hybrid_res) > 0, "Hybrid retriever returned no results"
    assert hybrid_res[0].chunk.entity == "NVDA", f"Expected NVDA chunk, got: {hybrid_res[0].chunk.entity}"
    logger.info(f"[PASS] Hybrid RAG Verified! Citation: {hybrid_res[0].citation}")

    # 4. Test Semantic Cache
    logger.info("--> [4/5] Verifying Semantic Caching...")
    cache = SemanticCache(similarity_threshold=0.85)
    cache.put("What are the key supply chain risks for NVDA and TSMC?", "Sample cached memo regarding TSMC risks.")
    hit = cache.get("What are the primary supply chain risks for NVDA with TSMC?")
    assert hit is not None, "Semantic cache failed to hit equivalent query"
    logger.info("[PASS] Semantic Cache Verified! Fast retrieval working!")

    # 5. Test Multi-Agent Supervisor
    logger.info("--> [5/5] Verifying Multi-Agent Supervisor Workflow...")
    risk_worker = RiskAnalystWorker(retriever=retriever)
    metric_worker = QuantitativeMetricWorker(query_engine=duck_adapter)
    supervisor = InvestmentSupervisorAgent(risk_worker=risk_worker, metric_worker=metric_worker)

    state = supervisor.run(
        user_query="So sánh rủi ro chuỗi cung ứng và tốc độ tăng trưởng doanh thu giữa NVDA và MSFT",
        target_entities=["NVDA", "MSFT"]
    )

    assert state.final_memo is not None and len(state.final_memo) > 100, "Supervisor failed to produce memo"
    assert len(state.audit_trail) >= 4, "Audit trail incomplete"
    logger.info("[PASS] Multi-Agent Supervisor Verified! Final Memo synthesized successfully!")
    logger.info("=================================================================")
    logger.info("[SUCCESS] ALL 5 ARCHITECTURAL TIERS PASSED VERIFICATION WITH 100% SUCCESS!")
    logger.info("=================================================================")


if __name__ == "__main__":
    run_all_checks()
