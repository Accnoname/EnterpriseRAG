"""
Multi-Agent Orchestration System:
1. RiskAnalystWorker: Qualitative Risk Factor extraction via Hybrid RAG.
2. QuantitativeMetricWorker: Exact Financial Metrics via DuckDB & Python Sandbox.
3. InvestmentSupervisorAgent: Decomposes goals, routes workers, critiques, and synthesizes memos.
"""

import os
import json
from typing import Dict, Any, List
from google import genai
from google.genai import types
from src.interfaces.ports import BaseAgentWorker, BaseVectorStore, BaseQueryEngine
from src.core.hybrid_retriever import HybridRetriever
from src.sandbox.runner import SafePythonSandbox
from src.contracts.schemas import FinancialTeamState
from src.common.config import DEFAULT_LLM_MODEL
from src.common.logger import get_logger

logger = get_logger("MultiAgentCore")


def _get_gemini_client():
    api_key = os.getenv("GEMINI_API_KEY")
    return genai.Client(api_key=api_key)


class RiskAnalystWorker(BaseAgentWorker):
    """Worker 1: Investigates business risks via Hybrid RAG."""

    def __init__(self, retriever: HybridRetriever):
        self.retriever = retriever

    def process_task(self, task_instruction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        entity = context.get("entity")
        query = task_instruction
        candidates = self.retriever.search_hybrid(query=query, top_k=3, filter_entity=entity)

        evidences = []
        for c in candidates:
            evidences.append({
                "citation": c.citation,
                "text": c.chunk.text[:400] + "...",
                "score": round(c.score, 4)
            })

        logger.info(f"Risk Analyst retrieved {len(evidences)} citations for {entity}")
        return {
            "entity": entity,
            "evidences": evidences,
            "summary_prompt": f"Identified {len(evidences)} risk disclosures with citations."
        }


class QuantitativeMetricWorker(BaseAgentWorker):
    """Worker 2: Runs exact SQL queries via DuckDB and calculations via Sandbox."""

    def __init__(self, query_engine: BaseQueryEngine):
        self.engine = query_engine
        self.sandbox = SafePythonSandbox()

    def process_task(self, task_instruction: str, context: Dict[str, Any]) -> Dict[str, Any]:
        entity = context.get("entity")
        # Run DuckDB query
        sql = f"SELECT year, revenue_billion, net_income_billion, operating_margin_pct FROM financial_metrics WHERE entity = '{entity}' ORDER BY year ASC"
        records = self.engine.execute_sql(sql)

        calculations = {}
        if len(records) >= 2:
            prev, curr = records[-2], records[-1]
            code = f"""
rev_growth = round(((curr_rev - prev_rev) / prev_rev) * 100, 2)
net_growth = round(((curr_net - prev_net) / prev_net) * 100, 2)
"""
            res = self.sandbox.execute(code, {
                "curr_rev": curr["revenue_billion"],
                "prev_rev": prev["revenue_billion"],
                "curr_net": curr["net_income_billion"],
                "prev_net": prev["net_income_billion"]
            })
            calculations = {
                "period": f"{prev['year']} -> {curr['year']}",
                "revenue_growth_pct": res.get("rev_growth"),
                "net_income_growth_pct": res.get("net_growth")
            }

        logger.info(f"Quantitative Worker processed metrics for {entity}")
        return {
            "entity": entity,
            "raw_metrics": records,
            "exact_calculations": calculations
        }


class InvestmentSupervisorAgent:
    """Supervisor: Decomposes tasks, coordinates workers, and synthesizes the Investment Memo."""

    def __init__(self, risk_worker: RiskAnalystWorker, metric_worker: QuantitativeMetricWorker):
        self.risk_worker = risk_worker
        self.metric_worker = metric_worker

    def run(self, user_query: str, target_entities: List[str]) -> FinancialTeamState:
        state = FinancialTeamState(user_query=user_query, target_entities=target_entities)
        state.audit_trail.append(f"Supervisor initiated workflow for entities: {target_entities}")

        all_risks = {}
        all_metrics = {}

        # 1. Execute workers for each entity
        for ent in target_entities:
            state.audit_trail.append(f"Dispatching RiskAnalystWorker for {ent}")
            risk_res = self.risk_worker.process_task(user_query, {"entity": ent})
            all_risks[ent] = risk_res

            state.audit_trail.append(f"Dispatching QuantitativeMetricWorker for {ent}")
            metric_res = self.metric_worker.process_task(user_query, {"entity": ent})
            all_metrics[ent] = metric_res

        state.qualitative_risks = all_risks
        state.quantitative_metrics = all_metrics

        # 2. Synthesize final Investment Memorandum using LLM
        client = _get_gemini_client()
        synthesis_prompt = f"""Bạn là Giám đốc Đầu tư (Chief Investment Officer). Hãy tổng hợp các dữ liệu định lượng (chính xác 100%) và rủi ro định tính dưới đây thành một bản Báo Cáo Chiến Lược Đầu Tư (Executive Investment Memorandum).

YÊU CẦU NGHIÊM NGẶT:
1. Mọi nhận định về rủi ro BẮT BUỘC phải đính kèm mã trích dẫn pháp lý [Citation].
2. Số liệu doanh thu, lợi nhuận, % tăng trưởng phải lấy đúng từ dữ liệu định lượng đã tính, KHÔNG đoán số.
3. Đưa ra so sánh khách quan và đề xuất hành động cụ thể cho ban lãnh đạo.

YÊU CẦU CỦA NHÀ ĐẦU TƯ:
"{user_query}"

DỮ LIỆU ĐỊNH LƯỢNG (TỪ DUCKDB & PYTHON SANDBOX):
{json.dumps(all_metrics, indent=2, ensure_ascii=False)}

DỮ LIỆU ĐỊNH TÍNH (TỪ HYBRID RAG TRÍCH DẪN):
{json.dumps(all_risks, indent=2, ensure_ascii=False)}
"""
        response = client.models.generate_content(
            model=DEFAULT_LLM_MODEL,
            contents=synthesis_prompt,
            config=types.GenerateContentConfig(temperature=0.2)
        )

        state.final_memo = response.text.strip()
        state.audit_trail.append("Supervisor successfully synthesized final Investment Memorandum.")
        return state

