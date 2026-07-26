"""
Minimal Streamlit UI for the Session 3 ADK agent (agent_rag.capstone_rag_agent).

Pattern copied from the ADK sample repo's streamlit_app.py: running an async ADK
Runner from a synchronous Streamlit callback via a dedicated event loop in a thread
(asyncio.run inside a ThreadPoolExecutor), since Streamlit callbacks are not async.

No API keys are read or displayed here -- agent_rag.py loads them from .env via
load_dotenv(), same as the CLI and the FastAPI app. This file never touches os.environ.

Run: streamlit run agent_ui.py
"""

import asyncio
import concurrent.futures
import os

import streamlit as st

import agent_rag

st.set_page_config(page_title="Capstone RAG Agent", page_icon="🤖")


def run_agent_sync(message: str, timeout: int = 120) -> dict:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, agent_rag.run_agent(message)).result(timeout=timeout)


st.title("Capstone RAG Agent")
st.caption(f"Agent: `{agent_rag.root_agent.name}` -- model `{agent_rag.MODEL}`")

if not os.getenv("GOOGLE_API_KEY"):
    st.warning("GOOGLE_API_KEY not set in .env -- the agent will fail to run.")

question = st.text_area(
    "User task / question",
    value="What are the key considerations for creating an AI policy?",
    height=80,
)

if st.button("Run agent", type="primary") and question.strip():
    with st.spinner("Running agent..."):
        result = run_agent_sync(question.strip())

    st.subheader("Final answer")
    st.success(result["answer"])

    st.subheader("Think -> Act -> Observe trace")
    if not result["trace"]:
        st.caption("No steps recorded.")
    for record in result["trace"]:
        label = {"THINK": "🧠 Think", "ACT": "🔧 Act", "OBSERVE": "👁 Observe", "FINAL": "✅ Final"}.get(
            record["type"], record["type"]
        )
        with st.expander(f"{label} ({record['author']})", expanded=(record["type"] != "OBSERVE")):
            st.code(record["detail"], language=None)
