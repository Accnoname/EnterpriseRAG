"""
Enterprise Financial AI - Interactive Conversational CLI.
Listens continuously, adapts to question complexity, retains conversation memory,
and executes exact SQL & Hybrid RAG tools on demand.
"""

import sys
import io
from pathlib import Path
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

# Force UTF-8 encoding
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stdin.reconfigure(encoding="utf-8")
    except Exception:
        pass

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.prompt import Prompt

from src.contracts.schemas import DocumentChunk, FinancialMetric
from src.adapters.duckdb_adapter import DuckDBAdapter
from src.adapters.chroma_adapter import ChromaAdapter
from src.core.hybrid_retriever import HybridRetriever
from src.agents.tools import AgentToolRegistry
from src.agents.conversational_agent import ConversationalFinancialAgent

console = Console()


def initialize_knowledge_base(duck: DuckDBAdapter, chroma: ChromaAdapter) -> HybridRetriever:
    metrics = [
        FinancialMetric(entity="NVDA", year=2023, revenue_billion=26.97, net_income_billion=4.37, operating_margin_pct=16.2),
        FinancialMetric(entity="NVDA", year=2024, revenue_billion=60.92, net_income_billion=29.76, operating_margin_pct=48.8),
        FinancialMetric(entity="MSFT", year=2023, revenue_billion=211.91, net_income_billion=72.36, operating_margin_pct=41.8),
        FinancialMetric(entity="MSFT", year=2024, revenue_billion=245.12, net_income_billion=88.14, operating_margin_pct=44.6),
        FinancialMetric(entity="AAPL", year=2023, revenue_billion=383.29, net_income_billion=96.99, operating_margin_pct=29.8),
        FinancialMetric(entity="AAPL", year=2024, revenue_billion=391.04, net_income_billion=93.74, operating_margin_pct=31.2),
    ]
    duck.load_metrics(metrics)

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
    chroma.index_chunks(chunks)
    return HybridRetriever(vector_store=chroma, chunks=chunks)


def main():
    console.print(Panel.fit(
        "[bold cyan]🏛️ ENTERPRISE FINANCIAL AI: INTERACTIVE CONVERSATIONAL AGENT[/bold cyan]\n"
        "[italic green]Lắng nghe liên tục • Ghi nhớ ngữ cảnh hội thoại • Phản hồi linh hoạt • 100% Chạy từ Ổ E[/italic green]",
        border_style="cyan"
    ))

    with console.status("[bold yellow]Khởi tạo Database Adapters & Mô hình AI Agent..."):
        duck = DuckDBAdapter()
        chroma = ChromaAdapter()
        retriever = initialize_knowledge_base(duck, chroma)
        tools = AgentToolRegistry(retriever=retriever, duckdb_adapter=duck)
        agent = ConversationalFinancialAgent(tool_registry=tools)

    console.print("[dim]Hệ thống đã sẵn sàng! Bạn có thể đặt bất kỳ câu hỏi nào (hỏi số, hỏi rủi ro, so sánh, hoặc trò chuyện nối tiếp).[/dim]")
    console.print("[bold yellow]Gợi ý:[/bold yellow] Gõ [bold cyan]'exit'[/bold cyan] hoặc [bold cyan]'quit'[/bold cyan] để thoát.\n")

    # Interactive REPL Loop
    while True:
        try:
            user_input = Prompt.ask("[bold green]👤 Bạn[/bold green]")
            if not user_input.strip():
                continue

            if user_input.lower().strip() in ["exit", "quit", "q"]:
                console.print("[bold yellow]Tạm biệt! Chúc bạn một ngày làm việc hiệu quả.[/bold yellow]")
                break

            with console.status("[bold magenta]AI Agent đang suy nghĩ và điều phối công cụ..."):
                response = agent.chat(user_input)

            console.print(Panel(
                Markdown(response),
                title="🤖 AI Financial Copilot",
                border_style="blue"
            ))
            console.print()

        except KeyboardInterrupt:
            console.print("\n[bold yellow]Đã dừng phiên làm việc.[/bold yellow]")
            break
        except Exception as e:
            console.print(f"[bold red]Lỗi: {e}[/bold red]")


if __name__ == "__main__":
    main()

