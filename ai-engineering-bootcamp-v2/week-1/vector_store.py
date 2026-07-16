"""Pinecone-backed vector store — embeds and retrieves text chunks for RAG."""

import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

load_dotenv(Path(__file__).resolve().parent / ".env")

# Same model at ingest and query time — otherwise similarity search is meaningless.
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSION = 1536

_openai_client = OpenAI()
_pinecone_client: Pinecone | None = None
_index = None


def _get_pinecone() -> Pinecone:
    global _pinecone_client
    if _pinecone_client is None:
        _pinecone_client = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    return _pinecone_client


def get_index():
    """Return the Pinecone index handle, creating the index on first use if needed."""

    global _index
    if _index is not None:
        return _index

    pc = _get_pinecone()
    index_name = os.environ["PINECONE_INDEX_NAME"]

    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(
                cloud=os.environ.get("PINECONE_CLOUD", "aws"),
                region=os.environ.get("PINECONE_REGION", "us-east-1"),
            ),
        )

    _index = pc.Index(index_name)
    return _index


def embed(text: str) -> list[float]:
    """Embed one piece of text with the shared embedding model."""

    response = _openai_client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    return response.data[0].embedding


def _chunk_id(document_id: str, chunk_index: int) -> str:
    # Derived from document_id + position so re-ingesting the same document overwrites
    # its old chunks instead of accumulating duplicates.
    return hashlib.sha256(f"{document_id}::{chunk_index}".encode("utf-8")).hexdigest()[:24]


def upsert_chunks(
    document_id: str,
    chunks: list[str],
    source: str | None = None,
    metadata: dict[str, str] | None = None,
    namespace: str | None = None,
) -> int:
    """Embed a document's chunks and store them with retrieval metadata."""

    index = get_index()
    vectors = []
    for chunk_index, chunk_text in enumerate(chunks):
        chunk_metadata = {
            "document_id": document_id,
            "chunk_index": chunk_index,
            "source": source or "",
            "text": chunk_text,
            **(metadata or {}),
        }
        vectors.append(
            {
                "id": _chunk_id(document_id, chunk_index),
                "values": embed(chunk_text),
                "metadata": chunk_metadata,
            }
        )
    index.upsert(vectors=vectors, namespace=namespace or "")
    return len(vectors)


def query_similar(question: str, top_k: int = 3, namespace: str | None = None) -> list[dict]:
    """Embed a question and return the top_k most similar stored chunks."""

    index = get_index()
    response = index.query(
        vector=embed(question),
        top_k=top_k,
        namespace=namespace or "",
        include_metadata=True,
    )
    return [
        {
            "id": match.id,
            "score": match.score,
            "text": (match.metadata or {}).get("text"),
            "document_id": (match.metadata or {}).get("document_id"),
            "chunk_index": (match.metadata or {}).get("chunk_index"),
            "source": (match.metadata or {}).get("source"),
        }
        for match in response.matches
    ]


def health_check() -> dict:
    """Confirm the vector store is reachable and report basic index stats."""

    index = get_index()
    stats = index.describe_index_stats()
    return {
        "reachable": True,
        "index_name": os.environ["PINECONE_INDEX_NAME"],
        "dimension": stats.dimension,
        "total_vector_count": stats.total_vector_count,
    }
