"""
agents/fraud_agent.py — Agent 3: Fraud Signal Detector

RUNS AFTER Agent 1 and 2 — reads both their analyses as context.
This is intentional: fraud patterns often only appear when you
cross-reference credit behaviour with income patterns.
"""

import json
from langchain_groq import ChatGroq
from agents.base import run_agent_with_retry
from models.state import LoanState


FRAUD_SYSTEM_PROMPT = """You are a fraud risk specialist at an Indian bank.

Your job is to detect fraud signals and inconsistencies in loan applications.
You cross-reference credit analysis, income analysis, and application remarks.

Policy fraud guidelines:
{policy_context}

LOOK FOR:
- Income declaration vs likely actual income inconsistencies
- Employment instability patterns that suggest fabrication
- Remarks that contradict structured data
- Round-number salaries, irregular employment patterns
- Any combination of factors from the credit + income analysis that raise suspicion

OUTPUT FORMAT (JSON only, no preamble):
{{
  "fraud_risk_level": "LOW|MEDIUM|HIGH|CRITICAL",
  "fraud_indicators": ["indicator1", "indicator2"],
  "inconsistencies_found": ["inconsistency1"],
  "remarks_analysis": "string — analysis of free-text remarks",
  "fraud_risk_subscore": number,
  "requires_manual_verification": boolean,
  "verification_items": ["item1"],
  "summary": "2-3 sentence fraud risk summary"
}}"""


def check_income_plausibility(annual_income: float,
                               employment_type: str,
                               employment_years: float) -> dict:
    """Tool: Check if declared income is plausible for profile."""
    # Rough income bands by employment type and experience
    if employment_type == "Salaried":
        expected_min = 300000 + (employment_years * 50000)
        expected_max = 500000 + (employment_years * 400000)
    elif employment_type in ["Self-Employed", "Business"]:
        expected_min = 400000
        expected_max = annual_income * 3  # wider band
    else:
        expected_min = 200000
        expected_max = 3000000

    is_round_number = annual_income % 100000 == 0

    return {
        "declared_income": annual_income,
        "expected_range": {"min": expected_min, "max": expected_max},
        "within_plausible_range": expected_min <= annual_income <= expected_max,
        "is_suspicious_round_number": is_round_number,
        "plausibility": (
            "PLAUSIBLE" if expected_min <= annual_income <= expected_max
            else "SUSPICIOUSLY_HIGH" if annual_income > expected_max
            else "SUSPICIOUSLY_LOW"
        ),
    }


def fraud_agent(state: LoanState, llm: ChatGroq, langfuse=None) -> LoanState:
    """
    Agent 3: Fraud Signal Detection Node.

    Reads:  application, policy_context, credit_analysis, income_analysis
    Writes: fraud_analysis, fraud_status
    """
    app = state["application"]

    plausibility = check_income_plausibility(
        app["annual_income"],
        app["employment_type"],
        app["employment_years"],
    )

    prompt = FRAUD_SYSTEM_PROMPT.format(
        policy_context=state.get("policy_context", "No policy context available.")
    )

    human_message = f"""Check this application for fraud signals:

APPLICATION DATA:
{json.dumps(app, indent=2)}

CREDIT ANALYSIS (Agent 1):
{state.get('credit_analysis', 'Not available')}

INCOME ANALYSIS (Agent 2):
{state.get('income_analysis', 'Not available')}

INCOME PLAUSIBILITY CHECK:
{json.dumps(plausibility, indent=2)}

Cross-reference all three sources. Flag any inconsistencies between
the structured data, the remarks, and the analysis findings.
Output valid JSON only."""
    
    # Create a span for this agent under the top-level trace
    trace = state.get("langfuse_trace")
    agent_span = None
    if trace:
        agent_span = trace.span(
            name="credit-agent",
            input={
                "credit_score":  state["application"]["credit_score"],
                "loan_amount":   state["application"]["requested_loan_amount"],
                "employment":    state["application"]["employment_type"],
            }
        )

    result = run_agent_with_retry(
        llm=llm,
        system_prompt=prompt,
        human_message=human_message,
        agent_name="fraud_agent",
        max_retries=2,
        trace=agent_span,
    )

    if result["status"] == "success":
        return {
            "fraud_analysis": result["content"],
            "fraud_status": "success",
            "current_step": "synthesizer",
        }
    else:
        return {
            "fraud_analysis": None,
            "fraud_status": "failed",
            "error_log": [result["error"]],
        }