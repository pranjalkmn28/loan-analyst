import operator
from typing import TypedDict, Optional, Annotated
from models.application import LoanApplication

class LoanState(TypedDict):
    # ── INPUT ──────────────────────────────────────────
    application: dict           # LoanApplication as dict

    # ── RAG CONTEXT (injected before agents run) ───────
    policy_context: Optional[str]   # Relevant RBI policy excerpts

    # ── AGENT OUTPUTS ──────────────────────────────────
    credit_analysis:  Optional[str]
    credit_status:    Optional[str]   # "success" | "failed"

    income_analysis:  Optional[str]
    income_status:    Optional[str]

    fraud_analysis:   Optional[str]
    fraud_status:     Optional[str]

    # ── FINAL OUTPUT ───────────────────────────────────
    risk_report:      Optional[dict]  # RiskReport as dict
    report_status:    Optional[str]

    # ── PIPELINE METADATA ──────────────────────────────
    # operator.add reducer lets parallel agents safely append errors concurrently
    error_log:        Annotated[list, operator.add]
    current_step:     Optional[str]
    langfuse_trace:   Optional[object]