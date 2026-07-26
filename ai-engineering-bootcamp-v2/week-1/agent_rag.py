"""
Session 3: minimal single-agent ADK job for the capstone RAG system.

Patterns copied from the adk-multi-agent-systems sample:
- Agent/Runner/InMemorySessionService wiring: demo1_routing.py
- Event-stream parsing (function_call / function_response / text): streamlit_app.py

Run: python agent_rag.py
"""

import asyncio
import logging

from dotenv import load_dotenv
from google.adk.agents import Agent
import vector_store
from google.adk.agents.invocation_context import LlmCallsLimitExceededError
from google.adk.agents.run_config import RunConfig
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("agent_rag")

MODEL = "gemini-flash-latest"  # gemini-2.5-flash is retired for new API keys as of this key

# Hard safety net: the loop cannot run forever even if the model misbehaves.
# One "LLM call" = one Think step, so this caps retrieve->think->retrieve->...->answer.
MAX_LLM_CALLS = 6

# --- Tools ---


def search_docs(query: str, top_k: int = 3) -> dict:
    """Search the capstone knowledge base for chunks relevant to a query.

    Real retrieval: embeds the query with OpenAI and queries the Pinecone index this
    capstone already ingested documents into (see vector_store.py). Use this whenever
    you need facts to answer a question -- never answer from memory.

    Returns {"status": "ok", "chunks": [...]} with each chunk's document_id, text, and
    similarity score, or {"error": ...} if the search itself failed (e.g. vector store
    unreachable) -- treat an error as "retry once or tell the user retrieval failed."
    """
    try:
        matches = vector_store.query_similar(query, top_k=top_k)
    except Exception as exc:
        return {"error": f"search_docs failed: {exc}"}

    if not matches:
        return {"status": "no_matches", "chunks": []}

    return {
        "status": "ok",
        "chunks": [
            {
                "document_id": match.get("document_id"),
                "text": match.get("text"),
                "score": match.get("score"),
            }
            for match in matches
        ],
    }


# --- Agent ---

root_agent = Agent(
    name="capstone_rag_agent",
    model=MODEL,
    description="Answers questions using the capstone's retrieved knowledge base context.",
    instruction=(
        "GOAL: Answer the user's question using ONLY context retrieved via search_docs.\n"
        "\n"
        "STEPS:\n"
        "1. Call search_docs with the user's question.\n"
        "2. If the returned chunks are insufficient, call search_docs again with a "
        "refined query -- at most twice total. If search_docs returns an error, you may "
        "retry once with the same query before giving up.\n"
        "3. Once you have enough context (or have exhausted retries), give a final answer.\n"
        "\n"
        "CONSTRAINTS:\n"
        "- Never answer from your own knowledge -- only from retrieved context.\n"
        "- Cite each source's document_id in parentheses, e.g. (POL-101).\n"
        "- Call search_docs at most twice per question (plus one retry on error).\n"
        "\n"
        'DONE means: you returned a final answer that is either grounded with citations, '
        'or -- if context was insufficient or search_docs kept failing -- exactly the '
        'string "I don\'t have enough information to answer that."'
    ),
    tools=[search_docs],
)

# --- Runner ---


def _truncate(text: str, limit: int = 200) -> str:
    return text if len(text) <= limit else text[:limit] + "..."


def _event_records(event) -> list[dict]:
    """Turn one ADK event into Think/Act/Observe/Final records -- mirrors the trace
    parsing in streamlit_app.py (author, function_call, function_response, text)."""

    author = getattr(event, "author", "?")
    content = getattr(event, "content", None)
    if not content or not content.parts:
        return []

    records = []
    for part in content.parts:
        fc = getattr(part, "function_call", None)
        fr = getattr(part, "function_response", None)
        text = getattr(part, "text", None)
        thought = getattr(part, "thought", False)

        if fc:
            records.append(
                {"type": "ACT", "author": author, "tool": fc.name,
                 "detail": f"{fc.name}({dict(fc.args or {})})"}
            )
        elif fr:
            records.append(
                {"type": "OBSERVE", "author": author, "tool": fr.name,
                 "detail": _truncate(str(fr.response))}
            )
        elif text and thought:
            records.append({"type": "THINK", "author": author, "detail": text})
        elif text and event.is_final_response():
            records.append({"type": "FINAL", "author": author, "detail": text})
    return records


def _log_event(event) -> list[dict]:
    """Log each record from this event and return them for callers that want the trace."""

    records = _event_records(event)
    for r in records:
        log.info("[%s] %-7s -> %s", r["author"], r["type"], r["detail"])
    return records


async def run_agent(message: str) -> dict:
    """Run capstone_rag_agent on one message.

    Returns {"answer": str, "steps": [...], "trace": [...]}:
    - steps: compact [{"tool": name, "observation": truncated str}] -- used by the FastAPI
      POST /agent endpoint in main.py.
    - trace: full Think/Act/Observe/Final records (see _event_records) -- used by the
      Streamlit UI (agent_ui.py) to render the whole reasoning sequence, not just tool calls.
    Shared by the CLI entrypoint (main, below), main.py, and agent_ui.py, so all three go
    through the exact same agent/tool/limit logic.
    """
    service = InMemorySessionService()
    runner = Runner(agent=root_agent, app_name="capstone_rag", session_service=service)
    session = await service.create_session(app_name="capstone_rag", user_id="user1")
    content = types.Content(role="user", parts=[types.Part(text=message)])

    run_config = RunConfig(max_llm_calls=MAX_LLM_CALLS)
    answer = "(no response)"
    trace: list[dict] = []
    try:
        async for event in runner.run_async(
            user_id="user1",
            session_id=session.id,
            new_message=content,
            run_config=run_config,
        ):
            trace.extend(_log_event(event))

            event_content = getattr(event, "content", None)
            if event.is_final_response() and event_content and event_content.parts:
                answer = event_content.parts[0].text
    except LlmCallsLimitExceededError:
        answer = f"(stopped: exceeded max_llm_calls={MAX_LLM_CALLS} without a final answer)"
        trace.append({"type": "FINAL", "author": "system", "detail": answer})

    steps = [{"tool": r["tool"], "observation": r["detail"]} for r in trace if r["type"] == "OBSERVE"]

    return {"answer": answer, "steps": steps, "trace": trace}


async def main():
    question = "What are the key considerations for creating an AI policy?"
    print(f"\nUser: {question}\n")
    result = await run_agent(question)
    print(f"\nAgent: {result['answer']}\n")


if __name__ == "__main__":
    asyncio.run(main())
