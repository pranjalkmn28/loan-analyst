"""
agents/credit_agent.py — Agent 1: Credit Pattern Analyst

WHAT IT DOES:
  Analyses credit score, credit history signals, and loan-to-income
  ratio against RBI policy guidelines retrieved via RAG.

TOOLS IT HAS:
  - analyse_credit_score: scores the credit rating per RBI bands
  - check_loan_to_income: validates LTI ratio against policy limits

KEY DESIGN:
  The agent receives policy_context from RAG — it doesn't rely on
  the LLM's training data for RBI rules. It reasons against actual
  retrieved policy text.
"""

import json
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from agents.base import run_agent_with_retry
from models.state import LoanState


CREDIT_SYSTEM_PROMPT = """You are a senior credit risk analyst at an Indian bank.

You analyse loan applications against RBI lending guidelines.

You have access to these policy guidelines:
{policy_context}

TOOLS AVAILABLE:
1. analyse_credit_score(score, employment_type) — evaluates credit score per RBI bands
2. check_loan_to_income(loan_amount, annual_income, loan_purpose) — checks LTI ratio

ANALYSIS REQUIREMENTS:
- Cite specific policy sections when making judgements
- Flag any credit concerns clearly
- Compute a credit risk sub-score (0-100, 0=lowest risk)
- Be specific — reference exact numbers from the application

OUTPUT FORMAT (JSON only, no preamble):
{{
  "credit_score_assessment": "string — assessment of the credit score",
  "lti_assessment": "string — assessment of loan-to-income ratio",
  "credit_flags": ["flag1", "flag2"],
  "credit_risk_subscore": number,
  "policy_sections_cited": ["section1", "section2"],
  "summary": "2-3 sentence overall credit assessment"
}}"""


def analyse_credit_score(score: int, employment_type: str) -> dict:
    """Tool: Evaluate credit score against RBI bands."""
    if score >= 750:
        return {"band": "EXCELLENT", "min_required": 650,
                "eligible": True, "note": "Eligible for preferential rates"}
    elif score >= 700:
        return {"band": "GOOD", "min_required": 650,
                "eligible": True, "note": "Standard approval process"}
    elif score >= 650:
        return {"band": "FAIR", "min_required": 650,
                "eligible": True, "note": "Additional verification required"}
    else:
        return {"band": "POOR", "min_required": 650,
                "eligible": False, "note": "Refer to senior underwriter"}


def check_loan_to_income(loan_amount: float, annual_income: float,
                          loan_purpose: str) -> dict:
    """Tool: Check loan-to-income ratio against policy limits."""
    lti = loan_amount / annual_income
    max_lti = 5.0 if "home" in loan_purpose.lower() else 3.0

    return {
        "lti_ratio": round(lti, 2),
        "max_allowed": max_lti,
        "within_limits": lti <= max_lti,
        "risk_level": (
            "LOW" if lti <= 3 else
            "MEDIUM" if lti <= 4 else
            "HIGH" if lti <= max_lti else
            "EXCEEDS_LIMIT"
        ),
    }


TOOLS = {
    "analyse_credit_score": analyse_credit_score,
    "check_loan_to_income": check_loan_to_income,
}


def credit_agent(state: LoanState, llm: ChatGroq, langfuse=None) -> LoanState:
    """
    Agent 1: Credit Pattern Analysis Node.

    Reads:  application, policy_context
    Writes: credit_analysis, credit_status
    """
    app = state["application"]
    prompt = CREDIT_SYSTEM_PROMPT.format(
        policy_context=state.get("policy_context", "No policy context available.")
    )

    # Pre-run tools — give the LLM computed facts, not raw numbers
    # This is more reliable than asking the LLM to call tools itself
    credit_score_result = analyse_credit_score(
        app["credit_score"], app["employment_type"]
    )
    lti_result = check_loan_to_income(
        app["requested_loan_amount"],
        app["annual_income"],
        app["loan_purpose"],
    )
    human_message = f"""Analyse the credit profile for this loan application:

Applicant: {app['applicant_name']}
Credit Score: {app['credit_score']}
Annual Income: ₹{app['annual_income']:,.0f}
Requested Loan: ₹{app['requested_loan_amount']:,.0f}
Loan Purpose: {app['loan_purpose']}
Employment: {app['employment_type']} — {app['employment_years']} years
Remarks: {app.get('remarks', 'None')}

TOOL RESULTS (already computed):
Credit Score Analysis: {json.dumps(credit_score_result, indent=2)}
Loan-to-Income Check: {json.dumps(lti_result, indent=2)}

Based on the policy guidelines and tool results above, provide your credit analysis.
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
        agent_name="credit_agent",
        max_retries=2,
        trace=agent_span,
    )
    if result["status"] == "success":
        return {
            "credit_analysis": result["content"],
            "credit_status": "success",
            "current_step": "income_agent",
        }
    else:
        return {
            "credit_analysis": None,
            "credit_status": "failed",
            "error_log": [result["error"]],
        }