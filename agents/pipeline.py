"""
agents/pipeline.py — LangGraph pipeline with parallel agent execution.

PARALLEL EXECUTION:
  LangGraph supports fan-out/fan-in natively.
  credit_agent and income_agent run simultaneously — no reason
  to wait for credit analysis before starting income analysis.

  Graph structure:
    inject_rag → [credit_agent ∥ income_agent] → fraud_agent → synthesizer

HOW PARALLEL WORKS IN LANGGRAPH:
  Add both nodes as edges from the same source node.
  LangGraph runs them concurrently and waits for both
  before proceeding to the next node.
"""


import os
import time
from functools import partial
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langfuse import Langfuse

from models.state import LoanState
from rag.retriever import retrieve_policy_context, build_policy_query
from agents.credit_agent import credit_agent
from agents.income_agent import income_agent
from agents.fraud_agent import fraud_agent
from agents.synthesizer import synthesizer_node


def build_pipeline(groq_api_key: str) -> tuple:
    """
    Returns (compiled_graph, langfuse_client).
    Langfuse client is shared across all requests.
    """

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        groq_api_key=groq_api_key,
        temperature=0,
        max_tokens=2048,
    )

    # ── Langfuse client ────────────────────────────────────────────────────
    # Graceful degradation: if keys not set, use a mock
    # so the pipeline works without Langfuse configured
    langfuse = _build_langfuse_client()

    credit     = partial(credit_agent,     llm=llm, langfuse=langfuse)
    income     = partial(income_agent,     llm=llm, langfuse=langfuse)
    fraud      = partial(fraud_agent,      llm=llm, langfuse=langfuse)
    synthesize = partial(synthesizer_node, llm=llm, langfuse=langfuse)

    graph = StateGraph(LoanState)

    graph.add_node("inject_rag",   _inject_rag_context)
    graph.add_node("credit_agent", credit)
    graph.add_node("income_agent", income)
    graph.add_node("fraud_agent",  fraud)
    graph.add_node("synthesizer",  synthesize)

    graph.set_entry_point("inject_rag")

    graph.add_edge("inject_rag",   "credit_agent")
    graph.add_edge("inject_rag",   "income_agent")
    graph.add_edge("credit_agent", "fraud_agent")
    graph.add_edge("income_agent", "fraud_agent")
    graph.add_edge("fraud_agent",  "synthesizer")
    graph.add_edge("synthesizer",  END)

    return graph.compile(), langfuse


def _build_langfuse_client():
    """
    Build real Langfuse client if keys exist.
    Fall back to a no-op mock if not configured.
    This way the pipeline never crashes due to missing Langfuse keys.
    """
    public_key  = os.getenv("LANGFUSE_PUBLIC_KEY", "")
    secret_key  = os.getenv("LANGFUSE_SECRET_KEY", "")

    if public_key and secret_key:
        print("✅ Langfuse tracing enabled.")
        return Langfuse(
            public_key=public_key,
            secret_key=secret_key,
            host="https://cloud.langfuse.com",
        )
    else:
        print("⚠️  Langfuse keys not set — tracing disabled. Pipeline still works.")
        return _MockLangfuse()


def _inject_rag_context(state: LoanState) -> LoanState:
    query   = build_policy_query(state["application"])
    context = retrieve_policy_context(query, k=6)
    return {**state, "policy_context": context, "current_step": "analysis"}


def should_run_fraud(state: LoanState) -> str:
    if state.get("credit_status") == "success" or state.get("income_status") == "success":
        return "fraud_agent"
    return "synthesizer"


def run_analysis(pipeline, langfuse, application: dict) -> dict:
    """
    Run the full pipeline.
    Creates one top-level Langfuse trace per loan application.
    Every agent span is a child of this trace.
    """
    # ── Top-level trace ────────────────────────────────────────────────────
    # This is the parent. Every agent creates child spans under it.
    # In Langfuse dashboard: one row per application, expand to see agents.
    trace = langfuse.trace(
        name="loan-analysis-pipeline",
        input={
            "applicant":    application.get("applicant_name"),
            "loan_amount":  application.get("requested_loan_amount"),
            "credit_score": application.get("credit_score"),
            "purpose":      application.get("loan_purpose"),
        },
        metadata={
            "pipeline_version": "1.0",
            "model": "llama-3.3-70b-versatile",
        }
    )

    t0 = time.time()

    initial_state: LoanState = {
        "application":    application,
        "policy_context": None,
        "credit_analysis": None, "credit_status": None,
        "income_analysis": None, "income_status": None,
        "fraud_analysis":  None, "fraud_status": None,
        "risk_report":     None, "report_status": None,
        "error_log":       [],
        "current_step":    "inject_rag",
        "langfuse_trace":  trace,   # pass trace through state
    }

    final_state = pipeline.invoke(initial_state)

    total_ms = int((time.time() - t0) * 1000)
    # ── Update top-level trace with final outcome ──────────────────────────
    report = final_state.get("risk_report") or {}
    trace.update(
        output={
            "decision":       report.get("decision"),
            "risk_score":     report.get("risk_score"),
            "risk_level":     report.get("risk_level"),
            "agents_completed": report.get("agents_completed"),
            "fallback_used":  report.get("fallback_used"),
            "total_ms":       total_ms,
            "errors":         final_state.get("error_log"),
        }
    )

    # Flush ensures all spans are sent before the response is returned
    langfuse.flush()
    return final_state


# ── Mock Langfuse — no-op when keys not configured ────────────────────────────
# Same interface as real Langfuse. Every method does nothing.
# This means zero code changes needed in agents when Langfuse is absent.

class _MockSpan:
    def end(self, **kwargs): pass
    def span(self, **kwargs): return _MockSpan()
    def update(self, **kwargs): pass

class _MockLangfuse:
    def trace(self, **kwargs): return _MockSpan()
    def flush(self): pass