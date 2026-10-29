"""
DuckDB Analytical Adapter: Executes in-process SQL over columnar data (Snappy Parquet).
"""

from typing import List, Dict, Any
from pathlib import Path
import duckdb
import pandas as pd
from src.interfaces.ports import BaseQueryEngine
from src.contracts.schemas import FinancialMetric
from src.common.config import GOLD_DIR
from src.common.logger import get_logger

logger = get_logger("DuckDBAdapter")


class DuckDBAdapter(BaseQueryEngine):
    def __init__(self, parquet_path: Path = GOLD_DIR / "financial_metrics.parquet"):
        self.parquet_path = parquet_path
        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(database=":memory:")
        self._ensure_table()

    def _ensure_table(self):
        if self.parquet_path.exists():
            self.con.execute(f"CREATE OR REPLACE TABLE financial_metrics AS SELECT * FROM read_parquet('{self.parquet_path}')")
            logger.info(f"Loaded existing metrics table from {self.parquet_path}")
        else:
            self.con.execute("""
                CREATE TABLE IF NOT EXISTS financial_metrics (
                    entity VARCHAR,
                    year INTEGER,
                    revenue_billion DOUBLE,
                    net_income_billion DOUBLE,
                    operating_margin_pct DOUBLE,
                    fcf_billion DOUBLE,
                    source VARCHAR
                )
            """)

    def load_metrics(self, metrics: List[FinancialMetric]) -> int:
        if not metrics:
            return 0
        df = pd.DataFrame([m.model_dump() for m in metrics])
        self.con.execute("CREATE OR REPLACE TABLE financial_metrics AS SELECT * FROM df")
        # Persist to Snappy Parquet
        df.to_parquet(self.parquet_path, compression="snappy", index=False)
        logger.info(f"Persisted {len(df)} financial metric records to {self.parquet_path}")
        return len(df)

    def execute_sql(self, sql_query: str) -> List[Dict[str, Any]]:
        """Executes read-only analytical SQL query."""
        logger.info(f"Executing SQL: {sql_query}")
        result = self.con.execute(sql_query).fetchdf()
        return result.to_dict(orient="records")

