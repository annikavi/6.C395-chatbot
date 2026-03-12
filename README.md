---
title: MIT Course Catalog Chatbot
emoji: 🎓
colorFrom: blue
colorTo: red
sdk: gradio
sdk_version: 5.23.3
python_version: "3.10"
app_file: app.py
pinned: false
secrets:
  - HF_TOKEN
---

# MIT Course Catalog Chatbot

An AI academic advisor that helps MIT students navigate the course catalog. Ask about distribution requirements (CI-H, HASS, REST, LAB), prerequisites, scheduling, course comparisons, and degree-specific electives.

## Architecture

**Data** — Course data is fetched from [Hydrant](https://hydrant.mit.edu) (Spring 2026 + Fall 2025), which provides clean, structured JSON with descriptions, prereqs, instructors, schedules, and distribution tags. Run `python -m src.scraper` to refresh `data/courses.json`.

**Retrieval** — `src/retriever.py` builds a BM25 index over all ~3000 courses. BM25 handles document length normalisation and term frequency saturation, which matters for matching short student queries against variable-length course descriptions.

**Two-pass RAG pipeline** (`src/chat.py`):
1. **Preflight** — A fast LLM call reads the student's message and outputs a JSON plan: which web searches to run (only for MIT institutional programs/terms not in the catalog), what BM25 queries to issue, and whether to filter by level (undergrad/grad).
2. **Gather** — Executes the plan: runs web searches for unfamiliar MIT programs (e.g. TPP, UROP, CRE), runs multiple BM25 queries against the catalog, and exact-matches any course numbers mentioned explicitly.
3. **Answer** — A second LLM call reasons over the gathered course excerpt and web context to generate the final response.

Web search uses Tavily if `TAVILY_API_KEY` is set, otherwise falls back to DuckDuckGo.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
HF_TOKEN=hf_your_token_here
TAVILY_API_KEY=tvly_your_key_here   # optional but recommended
```

Run locally:
```bash
python app.py
```

To refresh course data:
```bash
python -m src.scraper
```

## Deploying to HuggingFace Spaces

1. Create a new Space (Gradio SDK, CPU tier)
2. Push this repo to the Space remote
3. Add `HF_TOKEN` (and optionally `TAVILY_API_KEY`) as Space secrets under Settings
