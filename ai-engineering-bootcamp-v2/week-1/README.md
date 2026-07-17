# Week 1 — `/ask` Demo (5 stages)

Build a typed LLM endpoint step by step. Each stage is a standalone FastAPI app you can run and compare.

## Setup

```bash
cp .env.example .env          # OPENAI_API_KEY=sk-...
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Demo stages

| Stage | File | What you learn |
|-------|------|----------------|
| 1 | `serve_stage1.py` | Bare `/ask` — string answer + `tokens_used` |
| 2 | `serve_stage2.py` | Structured output via Pydantic + `completions.parse` |
| 3 | `serve_stage3.py` | Validation guardrail + retry (`force_bad` demo knob) |
| 4 | `serve_stage4.py` | Per-request `model` override + `latency_ms` |
| 5 | `serve_stage5.py` / `main.py` | Full system + `cost_usd` readout |

Run one stage at a time (only one server on port 8000):

```bash
uvicorn serve_stage1:app --host 127.0.0.1 --port 8000 --reload
# or the full system:
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

## Streamlit demo runner

Interactive UI for all five stages:

```bash
streamlit run demo_page.py
```

Open http://localhost:8501. Set **API base URL** to `http://127.0.0.1:8000` and start the matching stage server in another terminal.

## Test with curl

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG in one sentence?"}'
```

Stage 5 example (model + cost):

```bash
curl -s -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is chunking?", "model": "gpt-4o-mini"}'
```

## Smoke-test all stages

Requires `.venv` and a valid `OPENAI_API_KEY`:

```bash
python test_all_stages.py
```

## RAG golden-set eval

Run against the live Pinecone-backed `/ask` (retrieval + generation), verified manually against known documents (`handbook`, `POL-101`).

| # | Question | Expected Answer (short) | Retrieval Hit? | Faithful? | Correct? | Notes |
|---|---|---|:---:|:---:|:---:|---|
| 1 | How many days per week can employees work remotely? | Up to 3 days/week, with manager approval | ✅ | ✅ | ✅ | Answer matched expected almost verbatim, cited (POL-101). `sources_needed` flagged `true` despite a fully correct answer — known prompt inconsistency. |
| 2 | What are the core hours employees must be reachable on Slack when working remotely? | 10:00–15:00 | ✅ | ✅ | ✅ | Exact match, cited (POL-101). |
| 3 | What are the standard working hours at Northwind Robotics? | 09:00–17:30, Monday–Friday | ✅ | ✅ | ✅ | Exact match, cited (POL-101). |
| 4 | How many annual leave days do employees get? | 28 days plus public holidays | ✅ | ✅ | ✅ | Exact match, cited (POL-101). |
| 5 | What approval is required for fully remote work arrangements? | Director approval, reviewed every six months | ✅ | ✅ | ✅ | Exact match, cited (POL-101). |
| 6 *(refusal test)* | What is the company's parental leave policy? | *(should refuse — not in any ingested doc)* | N/A* | ✅ | ✅ | Model correctly said "I don't have enough information to answer that." instead of guessing, even though 5 irrelevant chunks were retrieved. |

\* Retrieval always returns top-5 nearest chunks regardless of relevance — there's no "correct" chunk to hit for an intentionally unanswerable question, so this is graded on whether the model correctly recognized the retrieved context was irrelevant (faithfulness), not on retrieval itself.

**Summary: 6/6 correct, 6/6 faithful, 5/5 retrieval hits on answerable questions, 1/1 correct refusal.**

## Project layout

```
week-1/
├── main.py              # Full system (stages 1–5 combined)
├── serve_stage1.py … serve_stage5.py
├── demo_page.py         # Streamlit test UI
├── test_all_stages.py   # Automated stage smoke tests
├── requirements.txt
├── .env.example
└── .gitignore
```
