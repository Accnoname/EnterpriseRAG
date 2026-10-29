"""
Interactive Enterprise Financial AI CLI Dashboard.
Powered by Multi-Agent Dual-Engine System (DuckDB + Hybrid RAG + Gemini 3.8 Flash).
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown

from src.contracts.schemas import DocumentChunk, FinancialMetric
from src.adapters.duckdb_adapter import DuckDBAdapter
from src.adapters.chroma_adapter import ChromaAdapter
from src.core.hybrid_retriever import HybridRetriever
from src.cache.semantic_cache import SemanticCache
from src.agents.supervisor import RiskAnalystWorker, QuantitativeMetricWorker, InvestmentSupervisorAgent

console = Console()


def initialize_demo_data(duck_adapter: DuckDBAdapter, chroma_adapter: ChromaAdapter) -> HybridRetriever:
    """Pre-seeds the Lakehouse with Magnificent 7 financial metrics and risk disclosures."""
    metrics = [
        FinancialMetric(entity="NVDA", year=2023, revenue_billion=26.97, net_income_billion=4.37, operating_margin_pct=16.2),
        FinancialMetric(entity="NVDA", year=2024, revenue_billion=60.92, net_income_billion=29.76, operating_margin_pct=48.8),
        FinancialMetric(entity="MSFT", year=2023, revenue_billion=211.91, net_income_billion=72.36, operating_margin_pct=41.8),
        FinancialMetric(entity="MSFT", year=2024, revenue_billion=245.12, net_income_billion=88.14, operating_margin_pct=44.6),
        FinancialMetric(entity="AAPL", year=2023, revenue_billion=383.29, net_income_billion=96.99, operating_margin_pct=29.8),
        FinancialMetric(entity="AAPL", year=2024, revenue_billion=391.04, net_income_billion=93.74, operating_margin_pct=31.2),
    ]
    duck_adapter.load_metrics(metrics)

    chunks = [
        DocumentChunk(
            chunk_id="NVDA_2024_01",
            entity="NVDA",
            period="2024",
            section="Item 1A - Risk Factors",
            text="We rely extensively on third-party foundries, primarily TSMC, to manufacture our semiconductor wafers. Any manufacturing bottleneck, geopolitical conflict, or supply disruption in Taiwan would directly threaten GPU supply and customer fulfillment."
        ),
        DocumentChunk(
            chunk_id="MSFT_2024_01",
            entity="MSFT",
            period="2024",
            section="Item 1A - Risk Factors",
            text="Our Azure Cloud and AI offerings depend on securing enormous power grid capacity and specialized AI computing accelerators. Fierce rivalry with Amazon AWS and Google Cloud could pressure margins."
        ),
        DocumentChunk(
            chunk_id="AAPL_2024_01",
            entity="AAPL",
            period="2024",
            section="Item 1A - Risk Factors",
            text="Substantial portions of our product manufacturing and component sourcing take place in mainland China. Regulatory shifts, trade tariffs, and antitrust inquiries against the App Store commission structure pose notable operational risks."
        )
    ]
    chroma_adapter.index_chunks(chunks)
    return HybridRetriever(vector_store=chroma_adapter, chunks=chunks)


def main():
    console.print(Panel.fit(
        "[bold cyan]🏛️ ENTERPRISE RAG: DUAL-ENGINE FINANCIAL MULTI-AGENT PLATFORM[/bold cyan]\n"
        "[italic green]Ports-and-Adapters Architecture | DuckDB OLAP + Hybrid RAG + Gemini 3.8 Flash[/italic green]",
        border_style="cyan"
    ))

    with console.status("[bold yellow]Initializing Database Adapters & Hybrid Retriever..."):
        duck = DuckDBAdapter()
        chroma = ChromaAdapter()
        retriever = initialize_demo_data(duck, chroma)
        cache = SemanticCache()

        risk_worker = RiskAnalystWorker(retriever=retriever)
        metric_worker = QuantitativeMetricWorker(query_engine=duck)
        supervisor = InvestmentSupervisorAgent(risk_worker=risk_worker, metric_worker=metric_worker)

    # Display Metrics Table
    table = Table(title="📊 Pre-Loaded Gold Layer: Snappy Columnar Financial Metrics")
    table.add_column("Entity / Ticker", style="cyan", justify="center")
    table.add_column("Year", style="white", justify="center")
    table.add_column("Revenue ($B)", style="green", justify="right")
    table.add_column("Net Income ($B)", style="magenta", justify="right")
    table.add_column("Op. Margin (%)", style="yellow", justify="right")

    for row in duck.execute_sql("SELECT entity, year, revenue_billion, net_income_billion, operating_margin_pct FROM financial_metrics ORDER BY entity, year"):
        table.add_row(row["entity"], str(row["year"]), f"${row['revenue_billion']:.2f}B", f"${row['net_income_billion']:.2f}B", f"{row['operating_margin_pct']:.1f}%")
    console.print(table)

    query = "So sánh rủi ro chuỗi cung ứng và tốc độ tăng trưởng doanh thu giữa NVDA và MSFT"
    console.print(f"\n[bold green]💡 Demo Query:[/bold green] [italic]'{query}'[/italic]\n")

    # Check Semantic Cache
    cached_response = cache.get(query)
    if cached_response:
        console.print("[bold yellow]⚡ [CACHE HIT][/bold yellow] Trả kết quả tức thì trong 5ms!")
        console.print(Panel(Markdown(cached_response), title="📄 Executive Investment Memorandum (Cached)"))
        return

    # Execute Multi-Agent Workflow
    with console.status("[bold magenta]Supervisor coordinating Workers (Risk Analyst + Quantitative Sandbox)..."):
        state = supervisor.run(user_query=query, target_entities=["NVDA", "MSFT"])

    # Audit Trail Table
    audit_tbl = Table(title="🔍 Multi-Agent Audit Trail & Orchestration Log")
    audit_tbl.add_column("Step", style="white", justify="center")
    audit_tbl.add_column("Agent Action", style="cyan")
    for idx, act in enumerate(state.audit_trail, start=1):
        audit_tbl.add_row(str(idx), act)
    console.print(audit_tbl)

    # Render Final Memo
    console.print(Panel(
        Markdown(state.final_memo or "No report synthesized."),
        title="📈 Executive Investment Memorandum (Synthesized with Citations & Exact Math)",
        border_style="green"
    ))

    # Save to Cache
    cache.put(query, state.final_memo or "")
    console.print("[dim green]✓ Successfully cached synthesized response into Semantic Cache.[/dim green]")


if __name__ == "__main__":
    main()
