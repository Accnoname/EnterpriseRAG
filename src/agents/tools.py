"""
Autonomous Multi-Agent Tools Definition:
Exposes tools directly for Gemini Function Calling:
1. `search_risk_factors(entity, query)` -> Hybrid RAG (BM25 + Dense Chroma)
2. `query_financial_database(sql_query)` -> DuckDB In-Process SQL
3. `calculate_in_sandbox(python_code, input_variables)` -> Safe Python Sandbox
"""

from typing import Dict, Any, List
from src.core.hybrid_retriever import HybridRetriever
from src.adapters.duckdb_adapter import DuckDBAdapter
from src.sandbox.runner import SafePythonSandbox
from src.common.logger import get_logger

logger = get_logger("AgentTools")


class AgentToolRegistry:
    def __init__(self, retriever: HybridRetriever, duckdb_adapter: DuckDBAdapter):
        self.retriever = retriever
        self.duckdb_adapter = duckdb_adapter
        self.sandbox = SafePythonSandbox()

    def search_risk_factors(self, entity: str, query: str) -> List[Dict[str, Any]]:
        """
        Tra cứu các rủi ro kinh doanh, chuỗi cung ứng, và pháp lý từ tài liệu báo cáo thường niên Form 10-K.
        Args:
            entity: Mã chứng khoán của công ty (ví dụ: 'NVDA', 'MSFT', 'AAPL').
            query: Từ khóa hoặc câu hỏi cần tra cứu rủi ro.
        Returns:
            Danh sách các trích dẫn và đoạn văn bản gốc có đánh dấu nguồn.
        """
        logger.info(f"🔧 [TOOL CALL] search_risk_factors for {entity} with query: '{query}'")
        candidates = self.retriever.search_hybrid(query=query, top_k=2, filter_entity=entity)
        results = []
        for c in candidates:
            results.append({
                "citation": c.citation,
                "text": c.chunk.text,
                "relevance_score": round(c.score, 4)
            })
        return results

    def query_financial_database(self, sql_query: str) -> List[Dict[str, Any]]:
        """
        Thực thi câu lệnh SQL để lấy số liệu tài chính thực tế từ bảng financial_metrics.
        Bảng financial_metrics bao gồm các cột:
        - entity (VARCHAR): Mã công ty (NVDA, MSFT, AAPL)
        - year (INTEGER): Năm tài chính (2023, 2024)
        - revenue_billion (DOUBLE): Doanh thu tính bằng tỷ USD
        - net_income_billion (DOUBLE): Lợi nhuận ròng tính bằng tỷ USD
        - operating_margin_pct (DOUBLE): Biên lợi nhuận hoạt động (%)
        Args:
            sql_query: Câu lệnh SELECT SQL hợp lệ.
        Returns:
            Danh sách các dòng kết quả dạng JSON.
        """
        logger.info(f"🔧 [TOOL CALL] query_financial_database SQL: '{sql_query}'")
        return self.duckdb_adapter.execute_sql(sql_query)

    def calculate_in_sandbox(self, python_code: str, input_variables: Dict[str, float]) -> Dict[str, Any]:
        """
        Chạy script Python trong môi trường Sandbox cô lập an toàn để tính toán các chỉ số tài chính chính xác 100%.
        Tuyệt đối không nhẩm tính số liệu.
        Args:
            python_code: Đoạn code tính toán (ví dụ: 'growth = round(((curr - prev) / prev) * 100, 2)').
            input_variables: Dictionary chứa các biến đầu vào cho phép tính.
        Returns:
            Dictionary chứa các biến kết quả tính toán.
        """
        logger.info(f"🔧 [TOOL CALL] calculate_in_sandbox Code: {python_code.strip()} with vars: {input_variables}")
        return self.sandbox.execute(python_code, input_variables)

