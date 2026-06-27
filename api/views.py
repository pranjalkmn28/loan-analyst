"""
api/views.py — Django views for the Loan Analyst API.

Two endpoints:
  POST /analyse   → runs the full pipeline, returns RiskReport
  GET  /health    → checks all components are ready

PRODUCTION PATTERNS IN THIS FILE:
  1. Pydantic validation at the boundary — bad input rejected before
     any LLM call is made. Saves tokens and latency.
  2. Timing — every response includes processing_time_ms.
     In production you'd alert if this exceeds a threshold.
  3. Explicit error taxonomy — validation errors (400) vs pipeline
     errors (500) vs service unavailable (503) are all distinct.
  4. Sanitization before validation — remarks are cleaned before
     Pydantic even sees them.
"""

import json
import time
import os
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from pydantic import ValidationError

from models.application import LoanApplication
from agents.pipeline import build_pipeline, run_analysis
from rag.ingestor import ingest_documents
from rag.retriever import retrieve_policy_context
from api.sanitizer import sanitize_remarks, sanitize_name

# ── Build pipeline once at startup ────────────────────────────────────────────
_pipeline = None
_langfuse = None

def get_pipeline():
    global _pipeline, _langfuse
    if _pipeline is None:
        groq_key = os.getenv("GROQ_API_KEY", "")
        if not groq_key:
            raise RuntimeError("GROQ_API_KEY not set")
        _pipeline, _langfuse = build_pipeline(groq_api_key=groq_key)
        print("✅ Loan Analyst pipeline ready.")
    return _pipeline, _langfuse

# Warm up pipeline at module load
try:
    get_pipeline()
except Exception as e:
    print(f"⚠️  Pipeline init failed: {e}")


# ── GET /health ────────────────────────────────────────────────────────────────
@require_http_methods(["GET"])
def health_view(request):
    from pathlib import Path
    chroma_ready = (Path("rag/chroma_db")).exists()

    return JsonResponse({
        "status": "ok",
        "pipeline_ready": _pipeline is not None,
        "groq_configured": bool(os.getenv("GROQ_API_KEY")),
        "langfuse_configured": bool(os.getenv("LANGFUSE_PUBLIC_KEY")),
        "rag_index_ready": chroma_ready,
    })


# ── POST /analyse ──────────────────────────────────────────────────────────────
@csrf_exempt
@require_http_methods(["POST"])
def analyse_view(request):
    start_time = time.time()

    # ── 1. Parse body ──────────────────────────────────────────────────────
    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON body."}, status=400)
    # ── 2. Sanitize free-text fields before validation ─────────────────────
    if "remarks" in body and body["remarks"]:
        remarks_result = sanitize_remarks(body["remarks"])
        if not remarks_result.is_safe:
            return JsonResponse(
                {"error": f"Invalid remarks: {remarks_result.reason}"},
                status=400
            )
        body["remarks"] = remarks_result.cleaned_value

    if "applicant_name" in body:
        name_result = sanitize_name(body["applicant_name"])
        if not name_result.is_safe:
            return JsonResponse(
                {"error": f"Invalid name: {name_result.reason}"},
                status=400
            )
        body["applicant_name"] = name_result.cleaned_value

    # ── 3. Validate with Pydantic ──────────────────────────────────────────
    # This catches: wrong types, out-of-range values, missing fields
    # Before a single LLM token is spent
    try:
        application = LoanApplication(**body)
    except ValidationError as e:
        errors = [
            {"field": err["loc"][0], "message": err["msg"]}
            for err in e.errors()
        ]
        return JsonResponse({"error": "Validation failed", "details": errors}, status=400)

    # ── 4. Get pipeline ────────────────────────────────────────────────────
    try:
        pipeline, langfuse = get_pipeline()
    except RuntimeError as e:
        return JsonResponse({"error": str(e)}, status=503)

    # ── 5. Run the pipeline ────────────────────────────────────────────────
    try:
        final_state = run_analysis(pipeline, langfuse, application.model_dump())
    except Exception as e:
        return JsonResponse(
            {"error": f"Pipeline error: {str(e)}"},
            status=500
        )

    # ── 6. Check we got a report ───────────────────────────────────────────
    if not final_state.get("risk_report"):
        return JsonResponse(
            {
                "error": "Pipeline completed but no report generated.",
                "error_log": final_state.get("error_log", []),
            },
            status=500
        )

    # ── 7. Attach timing and return ────────────────────────────────────────
    processing_ms = int((time.time() - start_time) * 1000)
    report = final_state["risk_report"]
    report["processing_time_ms"] = processing_ms

    return JsonResponse({
        "status":       final_state.get("report_status"),
        "report":       report,
        "error_log":    final_state.get("error_log") or [],
        "agents": {
            "credit": final_state.get("credit_status"),
            "income": final_state.get("income_status"),
            "fraud":  final_state.get("fraud_status"),
        }
    })