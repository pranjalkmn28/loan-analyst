"""
agents/synthesizer.py — Final node: produces the structured RiskReport.

This is NOT another analysis agent. It's a structured output node.
It reads all three analyses and produces a validated Pydantic RiskReport.

KEY CONCEPT — why a separate synthesizer:
  Each agent produces domain-specific JSON with different schemas.
  The synthesizer's only job is to read all three and output ONE
  unified RiskReport that Pydantic validates.
  If the LLM output doesn't match the schema, we catch it here
  before it ever reaches the client.
"""

import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from agents.base import run_agent_with_retry, safe_parse_json
from models.state import LoanState
from models.report import RiskReport, Decision, RiskLevel


SYNTHESIZER_PROMPT = """You are the final decision authority for loan applications.

You receive three specialist analyses and must synthesize them into
a single, definitive risk assessment and lending decision.

DECISION RULES (apply strictly):
- Any CRITICAL fraud risk → REJECT
- Credit score < 650 → REJECT
- DTI > 50% → REJECT
- Multiple HIGH risks across agents → REFER_TO_UNDERWRITER
- Single HIGH risk or multiple MEDIUM → CONDITIONAL_APPROVE
- All LOW/MEDIUM with no flags → APPROVE

Risk score calculation:
- Weight credit subscore: 40%
- Weight income subscore: 35%
- Weight fraud subscore: 25%

OUTPUT: Valid JSON matching this exact schema, nothing else:
{
  "decision": "APPROVE|CONDITIONAL_APPROVE|REJECT|REFER_TO_UNDERWRITER",
  "risk_score": <0-100 integer>,
  "risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "flags": ["flag1", "flag2"],
  "credit_analysis": "summary of credit findings",
  "income_analysis": "summary of income findings",
  "fraud_signals": "summary of fraud findings",
  "policy_references": ["RBI Section X.Y", "Internal Policy Part Z"],
  "reasoning": "comprehensive 3-4 sentence final reasoning"
}"""


def synthesizer_node(state: LoanState, llm: ChatGroq, langfuse=None) -> LoanState:
    """
    Final synthesis node.

    Reads:  credit_analysis, income_analysis, fraud_analysis
    Writes: risk_report, report_status
    """

    # Build fallback context if any agent failed
    credit  = state.get("credit_analysis") or '{"summary": "Credit analysis unavailable", "credit_risk_subscore": 50}'
    income  = state.get("income_analysis") or '{"summary": "Income analysis unavailable", "income_risk_subscore": 50}'
    fraud   = state.get("fraud_analysis")  or '{"summary": "Fraud analysis unavailable", "fraud_risk_subscore": 30}'

    fallback_used = (
        state.get("credit_status") != "success" or
        state.get("income_status") != "success" or
        state.get("fraud_status") != "success"
    )

    app = state["application"]

    human_message = f"""Synthesize a final loan decision for:

Applicant: {app['applicant_name']}
Requested: ₹{app['requested_loan_amount']:,.0f} for {app['loan_purpose']}

CREDIT ANALYSIS:
{credit}

INCOME ANALYSIS:
{income}

FRAUD ANALYSIS:
{fraud}

{"⚠️  NOTE: One or more agents had errors. Use available data with caution." if fallback_used else ""}

Apply the decision rules strictly. Output valid JSON only."""

    result = run_agent_with_retry(
        llm=llm,
        system_prompt=SYNTHESIZER_PROMPT,
        human_message=human_message,
        agent_name="synthesizer",
        max_retries=3,   # more retries — this is the most critical node
    )

    if result["status"] == "success":
        try:
            report_dict = safe_parse_json(result["content"])

            # Validate with Pydantic — if this fails, we catch it below
            report = RiskReport(
                **report_dict,
                agents_completed=sum([
                    state.get("credit_status") == "success",
                    state.get("income_status") == "success",
                    state.get("fraud_status") == "success",
                ]),
                fallback_used=fallback_used,
            )

            return {
                "risk_report": report.model_dump(),
                "report_status": "success",
                "current_step": "done",
            }

        except Exception as e:
            # Pydantic validation failed — build a safe fallback report
            fallback_report = RiskReport(
                decision=Decision.REFER_TO_UNDERWRITER,
                risk_score=75,
                risk_level=RiskLevel.HIGH,
                flags=["System error — manual review required"],
                credit_analysis=str(state.get("credit_analysis", "N/A"))[:200],
                income_analysis=str(state.get("income_analysis", "N/A"))[:200],
                fraud_signals="Could not complete fraud analysis",
                policy_references=[],
                reasoning="Automated analysis encountered an error. Refer to underwriter.",
                fallback_used=True,
                agents_completed=0,
            )
            return {
                "risk_report": fallback_report.model_dump(),
                "report_status": "fallback",
                "error_log": [str(e)],
                "current_step": "done",
            }

    return {
        "risk_report": None,
        "report_status": "failed",
        "current_step": "done",
        "error_log": [result["error"]],
    }