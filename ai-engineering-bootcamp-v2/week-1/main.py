"""Week 1 live demo — five stages in one file, built up live in class."""

import time
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError

import vector_store

# Load .env from this folder so the key is found regardless of shell working directory.
_ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(_ENV_PATH)

# Reuse one client so TLS handshakes are not repeated on every request.
app = FastAPI()
client = OpenAI()  # Reads OPENAI_API_KEY from the environment; never hardcode keys.

# Stage 4 default — strong general model; swap at request time for the live demo.
DEFAULT_MODEL = "gpt-4o"

# Stage 5 — per-1K-token input/output USD (derived from OpenAI list prices).
MODEL_PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o": (0.0025, 0.01),
    "gpt-4o-mini": (0.00015, 0.0006),
    "o3-mini": (0.0011, 0.0044),
}


class Answer(BaseModel):
    """Structured model output — this is what turns a chatbot into a component."""

    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    sources_needed: bool  # RAG: true when the retrieved context was insufficient to answer.


class AskRequest(BaseModel):
    """Typed request body so bad input is rejected before we spend tokens."""

    question: str
    force_bad: bool = False  # Stage 3 demo knob — first attempt breaks schema on purpose.
    model: str | None = None  # Stage 4 — optional override to swap models live.


class AskResponse(BaseModel):
    """Typed response so callers always get the same shape back."""

    answer: Answer
    tokens_used: int
    model: str
    latency_ms: int
    cost_usd: float
    retrieved_chunk_ids: list[str]


RAG_TOP_K = 5

GROUNDING_PROMPT_TEMPLATE = """Answer using ONLY the context below.
If the context does not contain the answer, say:
"I don't have enough information to answer that."
Cite the document_id of each chunk you used, e.g. "... (POL-101)".

Context:
{retrieved_chunks}

Question: {question}"""


def build_grounding_prompt(question: str, chunks: list[dict]) -> str:
    """Build the RAG prompt: forces citation by document_id and a fixed refusal string."""

    if not chunks:
        retrieved_chunks = "(no relevant context found)"
    else:
        retrieved_chunks = "\n\n".join(
            f"[document_id={chunk['document_id']}] {chunk['text']}" for chunk in chunks
        )
    return GROUNDING_PROMPT_TEMPLATE.format(retrieved_chunks=retrieved_chunks, question=question)


def compute_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Turn real usage into dollars — same prompt, different model, different cost."""

    prices = MODEL_PRICES_PER_1K.get(model, MODEL_PRICES_PER_1K[DEFAULT_MODEL])
    input_per_1k, output_per_1k = prices
    return (prompt_tokens / 1000 * input_per_1k) + (completion_tokens / 1000 * output_per_1k)


def call_model_structured(prompt: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 2 center: OpenAI structured output forces exactly the Answer schema.
    Returns parsed answer plus token counts from billing metadata.
    """

    completion = client.chat.completions.parse(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format=Answer,
    )

    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError("Model returned no parseable structured output")

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return parsed, total, prompt_tokens, completion_tokens


def call_model_unsafe(prompt: str, model: str) -> tuple[Answer, int, int, int]:
    """
    Stage 3 demo path: free-form JSON call, then validate locally.
    The bad instruction makes confidence a string so Pydantic rejects it reliably.
    """

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": (
                    f"{prompt}\n\n"
                    "Reply with ONLY a JSON object using keys answer, confidence, sources_needed. "
                    "Set confidence to the string 'very high' (not a number)."
                ),
            }
        ],
    )

    raw = completion.choices[0].message.content or ""
    # Guardrail: refuse malformed output instead of passing it through to clients.
    answer = Answer.model_validate_json(raw)

    usage = completion.usage
    total = usage.total_tokens if usage else 0
    prompt_tokens = usage.prompt_tokens if usage else 0
    completion_tokens = usage.completion_tokens if usage else 0
    return answer, total, prompt_tokens, completion_tokens


@app.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """RAG: retrieve grounding context, then answer with structured output, guardrails,
    and cost visibility. Embedding cost is not included in cost_usd (only chat tokens are)."""

    model = body.model or DEFAULT_MODEL
    last_error: str | None = None

    # Retrieve once — not inside the retry loop, so a validation retry doesn't re-embed.
    retrieved_chunks = vector_store.query_similar(body.question, top_k=RAG_TOP_K)
    retrieved_chunk_ids = [chunk["id"] for chunk in retrieved_chunks]
    prompt = build_grounding_prompt(body.question, retrieved_chunks)

    # Stage 3: one retry keeps the logic legible while still protecting callers.
    for attempt in range(2):
        try:
            start = time.perf_counter()

            # First attempt with force_bad uses the unsafe path; retry uses structured output.
            use_bad_path = body.force_bad and attempt == 0
            if use_bad_path:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_unsafe(
                    prompt, model
                )
            else:
                answer, tokens_used, prompt_tokens, completion_tokens = call_model_structured(
                    prompt, model
                )

            latency_ms = int((time.perf_counter() - start) * 1000)
            cost_usd = compute_cost_usd(model, prompt_tokens, completion_tokens)

            return AskResponse(
                answer=answer,
                tokens_used=tokens_used,
                model=model,
                latency_ms=latency_ms,
                cost_usd=round(cost_usd, 6),
                retrieved_chunk_ids=retrieved_chunk_ids,
            )
        except (ValidationError, ValueError) as exc:
            last_error = str(exc)
            continue

    # Clean failure — never leak a half-parsed response to the client.
    raise HTTPException(
        status_code=502,
        detail=f"Model response failed schema validation after retry: {last_error}",
    )


class IngestRequest(BaseModel):
    """One document's text to chunk, embed, and index."""

    text: str
    document_id: str
    source: str | None = None  # e.g. original filename
    metadata: dict[str, str] | None = None  # any extra tags to store alongside each chunk
    chunk_size: int = 800
    chunk_overlap: int = 100


class IngestResponse(BaseModel):
    document_id: str
    chunks_indexed: int
    status: str


# curl -s -X POST http://127.0.0.1:8000/ingest \
#   -H "Content-Type: application/json" \
#   -d '{"text": "Full document text goes here...", "document_id": "doc-1", "source": "handbook.pdf"}'
@app.post("/ingest")
def ingest(body: IngestRequest) -> IngestResponse:
    """Chunk a document, embed each chunk, and upsert it into the Pinecone index."""

    if not body.text.strip():
        raise HTTPException(status_code=400, detail="text must not be empty")
    if not body.document_id.strip():
        raise HTTPException(status_code=400, detail="document_id must not be empty")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=body.chunk_size,
        chunk_overlap=body.chunk_overlap,
    )
    chunks = splitter.split_text(body.text)

    count = vector_store.upsert_chunks(
        document_id=body.document_id,
        chunks=chunks,
        source=body.source,
        metadata=body.metadata,
    )

    return IngestResponse(document_id=body.document_id, chunks_indexed=count, status="indexed")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 3


class SearchResult(BaseModel):
    id: str
    score: float
    text: str | None = None


@app.post("/search")
def search(body: SearchRequest) -> list[SearchResult]:
    """Embed the query and return the most similar stored chunks — no LLM call."""

    matches = vector_store.query_similar(body.query, top_k=body.top_k)
    return [SearchResult(**match) for match in matches]


class RetrieveResult(BaseModel):
    id: str
    score: float
    document_id: str | None = None
    chunk_index: int | None = None
    source: str | None = None
    text: str | None = None


# curl -s "http://127.0.0.1:8000/debug/retrieve?q=How%20many%20remote%20days&top_k=5"
@app.get("/debug/retrieve")
def debug_retrieve(q: str, top_k: int = 5) -> list[RetrieveResult]:
    """Embed q and return the top matching chunks with scores/metadata. No LLM call — for
    verifying retrieval quality before wiring generation into /ask."""

    if not q.strip():
        raise HTTPException(status_code=400, detail="q must not be empty")

    matches = vector_store.query_similar(q, top_k=top_k)
    return [RetrieveResult(**match) for match in matches]


@app.get("/health/vector-store")
def vector_store_health() -> dict:
    """Debug endpoint: confirm Pinecone is reachable and report index stats."""

    try:
        return vector_store.health_check()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Vector store unreachable: {exc}")
