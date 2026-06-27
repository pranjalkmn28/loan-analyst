from pydantic import BaseModel, field_validator, Field
from enum import Enum
from typing import Literal, Optional

class EmploymentType(str, Enum):
    SALARIED = "Salaried"
    SELF_EMPLOYED = "Self-Employed"
    BUSINESS = "Business"
    FREELANCE = "Freelance"


class LoanPurpose(str, Enum):
    HOME_PURCHASE  = "Home purchase"
    HOME_RENOVATION = "Home renovation"
    VEHICLE        = "Vehicle"
    EDUCATION      = "Education"
    BUSINESS       = "Business"
    PERSONAL       = "Personal"
    MEDICAL        = "Medical"


class LoanApplication(BaseModel):
    applicant_name:       str
    age:                  int       = Field(ge=18, le=70)
    annual_income:        float     = Field(gt=0, description="Annual income in INR")
    requested_loan_amount: float    = Field(gt=0)
    loan_purpose:         LoanPurpose
    credit_score:         int       = Field(ge=300, le=900)
    existing_emis:        float     = Field(ge=0, description="Monthly EMI obligations in INR")
    employment_type:      EmploymentType
    employment_years:     float     = Field(ge=0)
    remarks:              Optional[str] = Field(None, max_length=1000)

    @field_validator("age")
    @classmethod
    def age_at_loan_maturity(cls, current_age):
        # Standard 20yr loan — age at maturity shouldn't exceed 70
        if current_age + 20 > 70:
            raise ValueError("Applicant age is too high for standard loan tenure")
        return current_age
    

    @property
    def monthly_income(self) -> float:
        return self.annual_income / 12
    
    @property
    def dti_ratio(self) -> float:
        # Debt-to-income ratio based on existing EMIs only.
        return (self.existing_emis / self.annual_income) * 100
    
    @property
    def loan_to_income_ratio(self) -> float:
        return self.requested_loan_amount / self.annual_income
