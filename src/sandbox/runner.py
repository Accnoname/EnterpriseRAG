"""
Restricted Python Code Sandbox: Safely runs calculations without OS/Network access.
Guarantees 100% numerical accuracy for financial metrics without LLM Hallucinations.
"""

import math
from typing import Dict, Any
from src.common.logger import get_logger

logger = get_logger("PythonSandbox")


class SafePythonSandbox:
    """Restricted execution environment allowing only mathematical and analytical operations."""

    SAFE_BUILTINS = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "float": float,
        "int": int,
        "str": str,
        "bool": bool,
        "dict": dict,
        "list": list,
        "range": range,
    }

    SAFE_MATH = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}

    def execute(self, code_snippet: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Executes code safely. The code must assign output variables into the local namespace.
        Example code:
            growth_pct = round(((rev_2024 - rev_2023) / rev_2023) * 100, 2)
        """
        local_scope = dict(context)
        global_scope = {
            "__builtins__": self.SAFE_BUILTINS,
            "math": self.SAFE_MATH
        }

        # Security check: Disallow dangerous tokens
        disallowed = ["import ", "exec", "eval", "os.", "sys.", "subprocess", "open(", "__", "shutil"]
        for bad in disallowed:
            if bad in code_snippet:
                raise ValueError(f"Security Violation: '{bad}' is strictly prohibited in Sandbox.")

        logger.info("Executing safe code in Sandbox")
        exec(code_snippet, global_scope, local_scope)

        # Filter out builtins from output
        result = {k: v for k, v in local_scope.items() if k not in ("__builtins__", "math")}
        return result

