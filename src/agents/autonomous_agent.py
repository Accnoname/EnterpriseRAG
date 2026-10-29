"""
True Autonomous ReAct Multi-Agent Engine:
Gemini 3.8 Flash acts as the Autonomous Decision Engine using native Function Calling.
Features:
- Native ReAct Loop (Thought -> Tool Call -> Observation -> Reflection)
- Automatic Tool Selection without hardcoded loops
- Structured Executive Investment Memo Output
- Step-by-step Execution Tracing
"""

import os
import json
from typing import List, Dict, Any
from google import genai
from google.genai import types
from src.contracts.schemas import FinancialTeamState
from src.agents.tools import AgentToolRegistry
from src.common.config import DEFAULT_LLM_MODEL
from src.common.logger import get_logger

logger = get_logger("AutonomousAgentEngine")


class AutonomousFinancialAgent:
    def __init__(self, tool_registry: AgentToolRegistry):
        self.tool_registry = tool_registry
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _execute_tool(self, func_name: str, func_args: Dict[str, Any]) -> Any:
        """Dynamically dispatches tool execution to the AgentToolRegistry."""
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
        else:
            raise ValueError(f"Unknown Tool: {func_name}")

    def run(self, user_query: str, target_entities: List[str], max_steps: int = 8) -> FinancialTeamState:
        state = FinancialTeamState(user_query=user_query, target_entities=target_entities)
        logger.info(f"🚀 [AGENT STARTED] Initiating Autonomous ReAct loop for: '{user_query}'")
        state.audit_trail.append(f"Autonomous Agent received goal: {user_query}")

        system_instruction = """Bạn là Senior Financial Intelligence AI Agent kiêm Giám đốc Phân tích Đầu tư.
Nhiệm vụ của bạn là giải quyết câu hỏi của người dùng một cách tự chủ bằng cách sử dụng các công cụ (Tools) được cung cấp.

BẠN CÓ 3 CÔNG CỤ CHUYÊN DỤNG:
1. `query_financial_database(sql_query)`: Lấy các số liệu tài chính doanh thu, lợi nhuận, margin từ bảng `financial_metrics`.
2. `calculate_in_sandbox(python_code, input_variables)`: BẮT BUỘC dùng công cụ này để tính các tỷ lệ %, CAGR, tăng trưởng YoY. KHÔNG ĐƯỢC TỰ NHẨM TÍNH TOÁN!
3. `search_risk_factors(entity, query)`: Tra cứu các rủi ro hoạt động, chuỗi cung ứng từ báo cáo 10-K gốc.

QUY TẮC BẮT BUỘC TRONG BÁO CÁO CUỐI CÙNG (INVESTMENT MEMO):
1. Mọi con số tài chính phải lấy từ SQL và Sandbox, kèm công thức tính rõ ràng.
2. Mọi nhận định về rủi ro BẮT BUỘC phải trích dẫn mã trích dẫn pháp lý chính xác (ví dụ: [NVDA | 2024 | Item 1A - Risk Factors | ID:NVDA_2024_01]).
3. Trình bày báo cáo theo chuẩn chuyên nghiệp:
   - Tóm tắt điều hành (Executive Summary)
   - Phân tích số liệu định lượng (kèm bảng so sánh)
   - Đánh giá rủi ro định tính & chuỗi cung ứng
   - Đề xuất hành động chiến lược và khuyến nghị tỷ trọng danh mục.
"""

        tools_declaration = [
            self.tool_registry.search_risk_factors,
            self.tool_registry.query_financial_database,
            self.tool_registry.calculate_in_sandbox
        ]

        chat = self.client.chats.create(
            model=DEFAULT_LLM_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2,
                tools=tools_declaration
            )
        )

        step_count = 0
        current_prompt = f"Mục tiêu phân tích: {user_query}\nCác công ty mục tiêu cần đối chiếu: {target_entities}"

        while step_count < max_steps:
            step_count += 1
            logger.info(f"⚡ [STEP {step_count}] Agent Reasoning...")
            response = chat.send_message(current_prompt)

            # Check if Agent called any functions
            if not response.function_calls:
                logger.info(f"🎯 [AGENT COMPLETED] Synthesized final investment memorandum in {step_count} steps.")
                state.final_memo = response.text.strip()
                state.audit_trail.append(f"Agent finished analysis after {step_count} steps.")
                break

            # Handle Function Calling
            tool_responses = []
            for call in response.function_calls:
                fn_name = call.name
                fn_args = dict(call.args)
                logger.info(f"🤖 [THOUGHT & ACTION] Agent decided to call: `{fn_name}` with args: {fn_args}")
                state.audit_trail.append(f"Step {step_count}: Called tool `{fn_name}`({fn_args})")

                try:
                    tool_output = self._execute_tool(fn_name, fn_args)
                except Exception as e:
                    tool_output = {"error": str(e)}
                    logger.error(f"Tool execution failed: {e}")

                state.audit_trail.append(f"Step {step_count}: Tool `{fn_name}` returned data.")
                tool_responses.append(types.Part.from_function_response(
                    name=fn_name,
                    response={"result": tool_output}
                ))

            # Send tool observations back to Agent
            current_prompt = tool_responses

        if not state.final_memo:
            state.final_memo = "Agent reached max iteration limit without final synthesis."

        return state

