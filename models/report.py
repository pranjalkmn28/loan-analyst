from enum import Enum
from pydantic import BaseModel, Field
from typing import Optional

class RiskLevel(str, Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"

class Decision(str, Enum):
    APPROVE             = "APPROVE"
    CONDITIONAL_APPROVE = "CONDITIONAL_APPROVE"
    REJECT              = "REJECT"
    REFER_TO_UNDERWRITER = "REFER_TO_UNDERWRITER"


class RiskReport(BaseModel):
    """
    The final structured output of the entire pipeline.
    Pydantic validates every field before this leaves the server.
    If the LLM produces something invalid, we catch it here — not in production.
    """
    decision:           Decision
    risk_score:         int        = Field(ge=0, le=100, description="0=lowest risk, 100=highest")
    risk_level:         RiskLevel
    flags:              list[str]  = Field(default_factory=list)

    credit_analysis:    str        = Field(description="Agent 1 findings")
    income_analysis:    str        = Field(description="Agent 2 findings")
    fraud_signals:      str        = Field(description="Agent 3 findings")

    policy_references:  list[str]  = Field(default_factory=list,
                                           description="RBI/policy docs cited")
    reasoning:          str        = Field(description="Final synthesized reasoning")

    # Metadata
    processing_time_ms: Optional[int] = None
    agents_completed:   int = 0
    fallback_used:      bool = False
