"""
api/sanitizer.py — Input sanitization for loan applications.

Two layers:
  1. Remarks field — only free text field, highest injection risk
  2. String fields — names, employment type etc.

Structured numeric fields (credit_score, income) are safe —
Pydantic validates types before they reach here.
"""

import re
from dataclasses import dataclass

INJECTION_PATTERNS = [
    r"ignore\s+(previous|above|all)\s+(instructions?|prompts?)",
    r"you\s+are\s+now\s+(a\s+)?",
    r"(reveal|dump|expose)\s+(your\s+)?(system\s+)?(prompt|instructions?)",
    r"new\s+(role|persona|instruction)",
    r"<\s*system\s*>",
    r"\[INST\]",
    r"###\s*instruction",
]

COMPILED = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]
MAX_REMARKS_LENGTH = 1000
MAX_NAME_LENGTH = 100


@dataclass
class SanitizeResult:
    is_safe: bool
    cleaned_value: str
    reason: str = ""


def sanitize_remarks(raw: str) -> SanitizeResult:
    if not raw:
        return SanitizeResult(is_safe=True, cleaned_value="")

    if len(raw) > MAX_REMARKS_LENGTH:
        return SanitizeResult(
            is_safe=False, cleaned_value="",
            reason=f"Remarks too long ({len(raw)} chars). Max {MAX_REMARKS_LENGTH}."
        )

    for pattern in COMPILED:
        if pattern.search(raw):
            return SanitizeResult(
                is_safe=False, cleaned_value="",
                reason="Remarks contain disallowed content."
            )

    # Strip control characters
    cleaned = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", raw)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return SanitizeResult(is_safe=True, cleaned_value=cleaned)


def sanitize_name(raw: str) -> SanitizeResult:
    if len(raw) > MAX_NAME_LENGTH:
        return SanitizeResult(is_safe=False, cleaned_value="",
                              reason="Name too long.")
    # Names should only have letters, spaces, dots, hyphens
    if not re.match(r"^[a-zA-Z\s.\-']+$", raw):
        return SanitizeResult(is_safe=False, cleaned_value="",
                              reason="Name contains invalid characters.")
    return SanitizeResult(is_safe=True, cleaned_value=raw.strip())