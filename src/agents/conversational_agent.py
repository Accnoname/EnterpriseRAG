"""
Conversational Multi-Turn Financial AI Agent Engine.
Features:
- Multi-turn Conversational Memory (Remembers previous queries and context).
- Adaptive Tone & Format (Short answers for fast facts, deep memos for comparisons, friendly explanations).
- Autonomous Tool Calling (DuckDB SQL, Hybrid RAG, Safe Python Sandbox).
- Automatic Model Fallback & Retry on 503 / 429 errors.
"""

import os
import time
from typing import List, Dict, Any, Optional
from google import genai
from google.genai import types
from google.genai.errors import ServerError, ClientError

from src.agents.tools import AgentToolRegistry
from src.common.config import PRIMARY_LLM_MODEL, FALLBACK_LLM_MODELS
from src.common.logger import get_logger

logger = get_logger("ConversationalAgent")


class ConversationalFinancialAgent:
    def __init__(self, tool_registry: AgentToolRegistry):
        self.tool_registry = tool_registry
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.chat_session = None
        self.active_model = PRIMARY_LLM_MODEL
        self._init_chat_session()

    def _init_chat_session(self):
        system_instruction = """Bạn là Trợ lý AI Tài chính & Kỹ sư Dữ liệu Doanh nghiệp thông minh, linh hoạt và biết lắng nghe.

PHONG CÁCH GIAO TIẾP VÀ TRẢ LỜI LINH HOẠT (ADAPTIVE RESPONSE):
1. **Hỏi nhanh / Tra cứu số đơn giản** (ví dụ: "Doanh thu Apple năm nay bao nhiêu?", "Nvidia lãi bao nhiêu?"):
   -> Trả lời ngắn gọn, trực diện, súc tích (1-3 câu), nêu rõ con số và nguồn dữ liệu. Không dài dòng văn tự.
2. **Hỏi so sánh / Đánh giá chiến lược** (ví dụ: "So sánh Nvidia và Microsoft", "Đánh giá triển vọng đầu tư"):
   -> Trình bày bài bản: Bảng so sánh chỉ số, phân tích rủi ro có trích dẫn, và khuyến nghị chiến lược rõ ràng.
3. **Trò chuyện / Hỏi tiếp câu trước (Follow-up)** (ví dụ: "Thế còn rủi ro thì sao?", "Tại sao lại như vậy?"):
   -> Tự động nhớ ngữ cảnh của các câu trước trong cuộc trò chuyện để trả lời liền mạch, tự nhiên như một đồng nghiệp chuyên gia tài chính.

BẠN CÓ 3 CÔNG CỤ (TOOLS) HỖ TRỢ:
- `query_financial_database(sql_query)`: Tra cứu số liệu doanh thu, lợi nhuận, margin từ bảng `financial_metrics` (AAPL, MSFT, NVDA).
- `calculate_in_sandbox(python_code, input_variables)`: Tính toán tỷ lệ %, CAGR, tăng trưởng chính xác 100%. Tuyệt đối không nhẩm tính số liệu.
- `search_risk_factors(entity, query)`: Tìm kiếm rủi ro từ báo cáo Form 10-K có kèm mã trích dẫn nguồn.

NGUYÊN TẮC:
- Mọi con số phải trích từ SQL/Sandbox.
- Mọi rủi ro pháp lý/chuỗi cung ứng phải có mã trích dẫn [Citation].
"""
        tools_declaration = [
            self.tool_registry.search_risk_factors,
            self.tool_registry.query_financial_database,
            self.tool_registry.calculate_in_sandbox
        ]

        logger.info(f"Initializing Chat Session with model: {self.active_model}")
        self.chat_session = self.client.chats.create(
            model=self.active_model,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                tools=tools_declaration
            )
        )

    def _execute_tool(self, func_name: str, func_args: Dict[str, Any]) -> Any:
        if func_name == "search_risk_factors":
            return self.tool_registry.search_risk_factors(
                entity=func_args.get("entity", ""),
                query=func_args.get("query", "")
            )
        elif func_name == "query_financial_database":
            return self.tool_registry.query_financial_database(
                sql_query=func_args.get("sql_query", "")
            )
        elif func_name == "calculate_in_sandbox":
            return self.tool_registry.calculate_in_sandbox(
                python_code=func_args.get("python_code", ""),
                input_variables=func_args.get("input_variables", {})
            )
        return {"error": f"Unknown tool: {func_name}"}

    def chat(self, user_message: str, max_steps: int = 6) -> str:
        """Sends a message in the continuous multi-turn session with ReAct function calling."""
        current_input = user_message
        step_count = 0

        while step_count < max_steps:
            step_count += 1
            try:
                response = self.chat_session.send_message(current_input)
            except (ServerError, ClientError) as e:
                logger.warning(f"Model {self.active_model} returned error: {e}. Attempting fallback...")
                for fb_model in FALLBACK_LLM_MODELS:
                    if fb_model != self.active_model:
                        self.active_model = fb_model
                        logger.info(f"Switching fallback model to: {self.active_model}")
                        self._init_chat_session()
                        time.sleep(1)
                        return self.chat(user_message, max_steps=max_steps)
                return "Hệ thống tạm thời quá tải từ phía nhà cung cấp API. Vui lòng thử lại sau giây lát."

            # If no tools called, we reached final answer
            if not response.function_calls:
                return response.text.strip()

            # Execute tool calls
            tool_responses = []
            for call in response.function_calls:
                fn_name = call.name
                fn_args = dict(call.args)
                logger.info(f"🤖 [TOOL] Agent calling `{fn_name}`: {fn_args}")

                try:
                    out = self._execute_tool(fn_name, fn_args)
                except Exception as err:
                    out = {"error": str(err)}

                tool_responses.append(types.Part.from_function_response(
                    name=fn_name,
                    response={"result": out}
                ))

            # Send tool outputs back to LLM
            current_input = tool_responses

        return "Agent đạt giới hạn số bước suy luận tối đa."
