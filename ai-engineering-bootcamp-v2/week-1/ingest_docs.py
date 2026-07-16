"""Batch-ingest every document in a folder into the RAG vector store via POST /ingest.

Usage: python ingest_docs.py [folder]  (defaults to the RAG_Sample_docs folder)
"""

import re
import sys
from pathlib import Path

import httpx
from pypdf import PdfReader

API_BASE = "http://127.0.0.1:8000"
DOCS_DIR = Path(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Saye_\AI-Internship\RAG_Sample_docs")


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def extract_text(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    return path.read_text(encoding="utf-8")


def main() -> None:
    files = sorted(
        p for p in DOCS_DIR.iterdir() if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md"}
    )

    for path in files:
        document_id = slugify(path.stem)  # stable — same filename always maps to the same ID
        text = extract_text(path)
        response = httpx.post(
            f"{API_BASE}/ingest",
            json={"text": text, "document_id": document_id, "source": path.name},
            timeout=300.0,
        )
        response.raise_for_status()
        data = response.json()
        print(f"{path.name} -> document_id={data['document_id']} chunks_indexed={data['chunks_indexed']}")

    health = httpx.get(f"{API_BASE}/health/vector-store", timeout=30.0).json()
    print(f"\nTotal chunks in vector store: {health['total_vector_count']}")


if __name__ == "__main__":
    main()
