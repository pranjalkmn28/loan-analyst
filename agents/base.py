"""
agents/base.py — Shared utilities for all agents.

run_agent_with_retry: wraps every LLM call with retry logic.
This is production thinking — LLMs fail occasionally (rate limits,
malformed output, network blips). Retrying once silently recovers
from most transient failures without the user ever knowing.
"""

import json
import time
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage


def safe_parse_json(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else cleaned
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    return json.loads(cleaned.strip())


def run_agent_with_retry(
    llm: ChatGroq,
    system_prompt: str,
    human_message: str,
    agent_name: str,
    max_retries: int = 2,
    trace=None,          # ← Langfuse trace passed in from pipeline
) -> dict:
    """
    Calls the LLM with retry logic + Langfuse span tracking.

    Each attempt creates a child span on the trace.
    You see in Langfuse: how many retries happened, which attempt
    succeeded, latency per attempt, full input/output.
    """
    last_error = None

    for attempt in range(max_retries + 1):

        # ── Langfuse: create a span for this attempt ───────────────────
        span = None
        if trace:
            span = trace.span(
                name=f"{agent_name}-attempt-{attempt + 1}",
                input={
                    "agent": agent_name,
                    "attempt": attempt + 1,
                    "prompt_length": len(human_message),
                }
            )

        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(
                    content=human_message if attempt == 0
                    else human_message + "\n\nIMPORTANT: Output ONLY valid JSON. No explanation, no markdown fences."
                ),
            ]

            t0 = time.time()
            response = llm.invoke(messages)
            latency_ms = int((time.time() - t0) * 1000)

            raw = response.content.strip()

            # Validate JSON
            safe_parse_json(raw)

            # ── Langfuse: log success ──────────────────────────────────
            if span:
                span.end(output={
                    "status": "success",
                    "latency_ms": latency_ms,
                    "output_length": len(raw),
                    # Log token usage if available
                    "tokens": getattr(response, "usage_metadata", None),
                })

            return {"status": "success", "content": raw}

        except json.JSONDecodeError as e:
            last_error = f"{agent_name}: JSON parse error on attempt {attempt + 1} — {str(e)}"
            if span:
                span.end(output={"status": "json_error", "error": last_error})
            if attempt < max_retries:
                time.sleep(1)

        except Exception as e:
            last_error = f"{agent_name}: Error on attempt {attempt + 1} — {str(e)}"
            if span:
                span.end(output={"status": "error", "error": last_error})
            if attempt < max_retries:
                time.sleep(2)

    return {"status": "failed", "error": last_error}