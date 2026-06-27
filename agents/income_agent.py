"""
agents/income_agent.py — Agent 2: Income Consistency Checker

WHAT IT DOES:
  Checks DTI ratio, employment stability, income adequacy
  against RBI policy guidelines.

RUNS IN PARALLEL with credit_agent — they are independent checks.
"""

import json
from langchain_groq import ChatGroq
from agents.base import run_agent_with_retry
from models.state import LoanState


INCOME_SYSTEM_PROMPT = """You are a senior income verification specialist at an Indian bank.

You assess income consistency, employment stability, and debt serviceability.

Policy guidelines for your reference:
{policy_context}

ANALYSIS REQUIREMENTS:
- Calculate and assess the DTI ratio carefully
- Flag employment stability concerns per RBI Section 3
- Assess whether income supports the requested EMI burden
- Compute an income risk sub-score (0-100, 0=lowest risk)

OUTPUT FORMAT (JSON only, no preamble):
{{
  "dti_assessment": "string",
  "employment_stability_assessment": "string",
  "income_adequacy_assessment": "string",
  "income_flags": ["flag1", "flag2"],
  "income_risk_subscore": number,
  "policy_sections_cited": ["section1"],
  "summary": "2-3 sentence overall income assessment"
}}"""


def check_dti_ratio(monthly_income: float, existing_emis: float,
                    proposed_emi_estimate: float) -> dict:
    """Tool: Check debt-to-income ratio."""
    total_obligations = existing_emis + proposed_emi_estimate
    dti = (total_obligations / monthly_income) * 100
    housing_ratio = (proposed_emi_estimate / monthly_income) * 100

    return {
        "current_dti": round(dti, 1),
        "housing_expense_ratio": round(housing_ratio, 1),
        "max_dti_allowed": 50.0,
        "max_housing_ratio_allowed": 35.0,
        "dti_status": (
            "ACCEPTABLE" if dti <= 40 else
            "ELEVATED" if dti <= 50 else
            "EXCEEDS_LIMIT"
        ),
        "within_limits": dti <= 50,
    }


def assess_employment_stability(employment_years: float,
                                  employment_type: str) -> dict:
    """Tool: Assess employment stability against RBI requirements."""
    min_required = {
        "Salaried": 2.0,
        "Self-Employed": 3.0,
        "Business": 3.0,
        "Freelance": 2.0,
    }.get(employment_type, 2.0)

    current_employer_months = employment_years * 12

    return {
        "total_years": employment_years,
        "min_required_years": min_required,
        "meets_minimum": employment_years >= min_required,
        "current_employer_months": round(current_employer_months, 1),
        "recent_job_change": current_employer_months < 6,
        "stability_rating": (
            "STABLE" if employment_years >= min_required and
                         current_employer_months >= 6
            else "MODERATE" if employment_years >= min_required
            else "UNSTABLE"
        ),
    }


def income_agent(state: LoanState, llm: ChatGroq, langfuse=None) -> LoanState:
    """
    Agent 2: Income Consistency Node — runs parallel to credit_agent.

    Reads:  application, policy_context
    Writes: income_analysis, income_status
    """
    app = state["application"]
    monthly_income = app["annual_income"] / 12

    # Estimate proposed EMI (rough: loan / 240 months at ~8.5%)
    proposed_emi_estimate = app["requested_loan_amount"] * 0.00876

    dti_result = check_dti_ratio(
        monthly_income, app["existing_emis"], proposed_emi_estimate
    )
    employment_result = assess_employment_stability(
        app["employment_years"], app["employment_type"]
    )

    prompt = INCOME_SYSTEM_PROMPT.format(
        policy_context=state.get("policy_context", "No policy context available.")
    )

    human_message = f"""Assess income and employment for this application:

Applicant: {app['applicant_name']}
Annual Income: ₹{app['annual_income']:,.0f}
Monthly Income: ₹{monthly_income:,.0f}
Existing EMIs: ₹{app['existing_emis']:,.0f}/month
Requested Loan: ₹{app['requested_loan_amount']:,.0f}
Employment: {app['employment_type']} — {app['employment_years']} years
Remarks: {app.get('remarks', 'None')}

TOOL RESULTS:
DTI Analysis: {json.dumps(dti_result, indent=2)}
Employment Stability: {json.dumps(employment_result, indent=2)}
Estimated Proposed EMI: ₹{proposed_emi_estimate:,.0f}/month

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
        agent_name="income_agent",
        max_retries=2,
        trace=agent_span,
    )

    if result["status"] == "success":
        return {
            "income_analysis": result["content"],
            "income_status": "success",
        }
    else:
        return {
            "income_analysis": None,
            "income_status": "failed",
            "error_log": [result["error"]],
        }