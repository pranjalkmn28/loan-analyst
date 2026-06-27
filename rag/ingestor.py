"""
rag/ingestor.py — Document ingestion pipeline.

Reads policy documents → chunks them → embeds → stores in ChromaDB.

Run this ONCE to build the vector store.
After that, retriever.py queries it on every loan application.

WHY THESE CHUNK SETTINGS:
- chunk_size=500: Small enough that each chunk covers one topic
  (e.g. one section on DTI rules). Too large = retrieved chunks
  have noise. Too small = missing context.
- chunk_overlap=50: Prevents a sentence being cut mid-thought
  at a chunk boundary. 10% overlap is standard.

WHY sentence-transformers (not OpenAI embeddings):
- Free, runs locally, no API calls for embedding
- all-MiniLM-L6-v2 is small (80MB) but good for semantic search
- In production: switch to text-embedding-3-small for better accuracy
"""


import os
from pathlib import Path
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

DOCS_DIR    = Path(__file__).parent / "documents"
CHROMA_DIR  = Path(__file__).parent / "chroma_db"
COLLECTION  = "loan_policy"

def get_embeddings():
    """
    Free local embeddings using sentence-transformers.
    Downloads ~80MB model on first run, cached after that.
    """
    return HuggingFaceEmbeddings(
        model_name = "all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def ingest_documents(force_reingest: bool = False) -> int:
    """
    Load all documents from rag/documents/, chunk them,
    embed them, and store in ChromaDB.

    Returns the number of chunks stored.
    force_reingest: if True, clears existing DB and rebuilds.
    """

    # Skip if already ingested (don't re-embed on every server start)
    if CHROMA_DIR.exists() and not force_reingest:
        print(f"✅ ChromaDB already exists at {CHROMA_DIR}. Skipping ingest.")
        print("   Pass force_reingest=True to rebuild.")
        return 0
    
    print("📄 Loading policy documents...")
    documents = []

    for file_path in DOCS_DIR.iterdir():
        if file_path.suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            docs = loader.load()
            # Tag each doc with its source filename
            for doc in docs:
                doc.metadata["source"] = file_path.name
            documents.extend(docs)
            print(f"   Loaded: {file_path.name} ({len(docs)} document(s))")
        
        elif file_path.suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            docs = loader.load()
            for doc in docs:
                doc.metadata["source"] = file_path.name
            documents.extend(docs)
            print(f"   Loaded: {file_path.name} ({len(docs)} page(s))")
    
    if not documents:
        raise ValueError(f"No documents found in {DOCS_DIR}")

    # ── Chunk the documents ───────────────────────────────────────────────
    print("\n✂️  Chunking documents...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        # Split on section breaks first, then paragraphs, then sentences
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.split_documents(documents)
    print(f"   {len(documents)} document(s) → {len(chunks)} chunks")

    # ── Embed and store ───────────────────────────────────────────────────
    print("\n🔢 Embedding chunks (first run downloads ~80MB model)...")
    embeddings = get_embeddings()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION,
    )

    print(f"\n✅ Ingestion complete. {len(chunks)} chunks stored in {CHROMA_DIR}")
    return len(chunks)


if __name__ == "__main__":
    ingest_documents(force_reingest=True)