"""
rag/retriever.py — Query the vector store.

Called before every agent run to fetch relevant policy context.
The returned string gets injected into agent prompts as {policy_context}.

WHY k=6:
  Each chunk is ~500 words. k=6 gives ~3000 words of policy context.
  Enough to cover credit + income + fraud rules without bloating the prompt.
  In production: tune k based on your average prompt token budget.
"""

from pathlib import Path
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_DIR = Path(__file__).parent / "chroma_db"
COLLECTION = "loan_policy"

# Module-level cache — load once, reuse across requests
_vectorstore = None


def _get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        if not CHROMA_DIR.exists():
            raise RuntimeError(
                "ChromaDB not found. Run: python -m rag.ingestor"
            )
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        _vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings,
            collection_name=COLLECTION,
        )
    return _vectorstore


def retrieve_policy_context(query: str, k: int = 6) -> str:
    """
    Given a natural language query about a loan application,
    retrieve the most relevant policy chunks.

    Returns a formatted string ready to inject into an agent prompt.

    The query is constructed from the loan application details —
    not from user input — so no prompt injection risk here.
    """

    vectorstore = _get_vectorstore()
    docs = vectorstore.similarity_search(query, k=k)

    if not docs:
        return "No specific policy guidelines retrieved. Apply general lending norms."
    
    # Format with source citations — agents will reference these
    sections = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "Unknown")
        sections.append(
            f"[Policy Excerpt {i} — Source: {source}]\n{doc.page_content}"
        )

    return "\n\n---\n\n".join(sections)


def build_policy_query(application: dict) -> str:
    """
    Construct a targeted retrieval query from the loan application.
    Better query = more relevant chunks retrieved.

    This is a key design decision:
    A generic query like "loan guidelines" retrieves generic chunks.
    A specific query retrieves exactly the rules that apply to THIS applicant.
    """
    parts = []

    credit_score = application.get("credit_score", 0)
    if credit_score < 650:
        parts.append("credit score below 650 rejection criteria")
    elif credit_score < 700:
        parts.append("credit score 650-699 additional verification")
    else:
        parts.append("credit score above 700 approval")

    emp_years = application.get("employment_years", 0)
    if emp_years < 0.5:
        parts.append("recent job change less than 6 months employment verification")

    dti = (application.get("existing_emis", 0) /
           (application.get("annual_income", 1) / 12)) * 100
    if dti > 40:
        parts.append("high debt to income ratio DTI above 40 percent")

    lti = (application.get("requested_loan_amount", 0) /
           application.get("annual_income", 1))
    if lti > 4:
        parts.append("loan to income ratio above 4x home loan limits")

    emp_type = application.get("employment_type", "")
    if emp_type in ["Self-Employed", "Freelance"]:
        parts.append("self employed income verification ITR requirements")

    parts.append("fraud detection income verification employment")

    return " | ".join(parts)

