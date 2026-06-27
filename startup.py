"""
startup.py — Runs on every Railway deploy.

Ingests policy documents into ChromaDB if not already done.
On Railway, filesystem resets on deploy so we always reingest.
Locally, skip if already ingested (saves time).
"""

import os
import sys

def main():
    is_production = os.getenv("RAILWAY_ENVIRONMENT") is not None

    print("🚀 Starting Loan Analyst API...")

    # Always reingest on Railway (fresh filesystem each deploy)
    # Skip locally if already ingested
    from rag.ingestor import ingest_documents
    ingest_documents(force_reingest=is_production)

    print("✅ Startup complete.")

if __name__ == "__main__":
    main()