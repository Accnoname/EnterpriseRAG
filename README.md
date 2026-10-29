# 🏛️ EnterpriseRAG: Pluggable Dual-Engine Financial & Business Intelligence AI Platform

Enterprise-grade, domain-agnostic AI analytics platform implementing a **Dual-Engine Architecture** (Hybrid RAG + In-Process Analytical OLAP) with **Multi-Agent Orchestration (Supervisor-Worker Pattern)** and **Ports-and-Adapters (Hexagonal) Architecture**.

---

## 🌟 Core Highlights
1. **Ports-and-Adapters (Hexagonal Architecture):** Decoupled abstractions for Ingestion, Vector Storage, Query Engines, and Agents. Seamlessly swap datasets or vector databases via config.
2. **Dual-Engine Precision:**
   - **Engine 1 (Qualitative / Hybrid RAG):** BM25 lexical search + Dense vector embeddings with Reciprocal Rank Fusion (RRF) and Cross-Encoder Reranking for strict contextual grounding.
   - **Engine 2 (Quantitative / Analytical OLAP):** DuckDB in-process SQL execution over columnar Snappy Parquet with Sandboxed Python code calculation (Zero Hallucination on numerical metrics).
3. **Multi-Agent Supervisor Hierarchy:**
   - **Investment Supervisor Agent:** Coordinates tasks, manages state transitions, and enforces quality gates.
   - **Risk Analyst Agent:** Conducts qualitative business risk investigations with verifiable source citations.
   - **Quantitative Metric Agent:** Executes exact SQL queries and sandbox calculations for financial KPIs (YoY growth, operating margins, valuation).
4. **Enterprise Hardening:** Semantic Caching (sub-10ms response, $0 token spend on recurring queries), structured JSONL audit logs, and self-verifying test suite.

---

## 📐 Architecture Overview

```text
                               ┌────────────────────────────────────────────────────────┐
                               │                    ANY DATA SOURCE                     │
                               │  - SEC Form 10-K Filings / Financial Disclosures       │
                               │  - Internal Company Reports / Custom PDFs / CSVs       │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │                 DOMAIN DATA CONTRACTS                  │
                               │   - DocumentChunk (ID, Entity, Period, Text, Source)   │
                               │   - FinancialMetric (Entity, Year, Metric, Value)      │
                               │   - FinancialTeamState (Immutable Shared State)        │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                           ┌──────────────────────────────┴──────────────────────────────┐
                           │                                                             │
                           ▼                                                             ▼
        ┌────────────────────────────────────┐                        ┌────────────────────────────────────┐
        │       ENGINE 1: HYBRID RAG         │                        │      ENGINE 2: OLAP ANALYTICS      │
        │  <<BaseVectorStore Adapter>>       │                        │    <<BaseQueryEngine Adapter>>     │
        │  - ChromaDB / Qdrant Adapter       │                        │  - DuckDB Columnar SQL Engine      │
        │  - BM25 Okapi + MiniLM Dense       │                        │  - Snappy Compressed Parquet Store │
        │  - Reciprocal Rank Fusion (RRF)    │                        │  - Restricted Python Code Sandbox  │
        │  - Cross-Encoder Re-ranker         │                        │  - Exact Zero-Hallucination Math   │
        └─────────────────┬──────────────────┘                        └──────────────────┬─────────────────┘
                          │                                                              │
                          └──────────────────────────────┬───────────────────────────────┘
                                                         │
                                                         ▼
                               ┌────────────────────────────────────────────────────────┐
                               │            FINANCIAL MULTI-AGENT SUPERVISOR            │
                               │  - Task Decomposition & Routing                        │
                               │  - Qualitative Risk Analyst Worker                     │
                               │  - Quantitative Metrics Worker                         │
                               │  - Self-Critique & Quality Gate Review                 │
                               │  - Executive Investment Memo Synthesis                 │
                               └──────────────────────────┬─────────────────────────────┘
                                                          │
                                                          ▼
                               ┌────────────────────────────────────────────────────────┐
                               │                ENTERPRISE SERVING LAYER                │
                               │  - Semantic Cache (Redis / In-Memory Embedding Cache)  │
                               │  - JSONL Structured Audit Logging & Observability      │
                               │  - Interactive CLI Terminal Interface (Rich Formatting)│
                               └────────────────────────────────────────────────────────┘
```

---

## 🚀 Quickstart & One-Click Verification

Run the entire end-to-end self-verifying pipeline from scratch:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the end-to-end self-verification suite
python run_verification.py

# 3. Launch the Interactive Financial Multi-Agent Assistant
python main.py
```

---

## 📂 Repository Structure

```text
EnterpriseRAG/
├── src/
│   ├── contracts/          # Strict Pydantic V2 Domain Schemas
│   ├── interfaces/         # Pluggable Ports (BaseVectorStore, BaseQueryEngine)
│   ├── adapters/           # Concrete Adapters (ChromaDB, DuckDB, SEC)
│   ├── core/               # BM25, Dense Embeddings, RRF, Reranker
│   ├── sandbox/            # Restricted Python Execution Environment
│   ├── agents/             # Multi-Agent Supervisor & Workers
│   ├── cache/              # Semantic Caching Layer
│   └── common/             # Config & Structured Logger
├── data/                   # Medallion Lakehouse Tiers (Bronze, Silver, Gold, Curated)
├── tests/                  # Automated Unit & Integration Tests
├── run_verification.py     # Automated end-to-end self-testing runner
├── main.py                 # Interactive Rich CLI Dashboard
├── requirements.txt        # Production dependencies
└── README.md               # Architecture & Documentation
```

