---
title: 6.C395-chatbot
app_file: app.py
sdk: gradio
sdk_version: 6.9.0
---
# MIT Course Catalog Chatbot

An AI academic advisor that helps MIT students navigate the course catalog. Ask about distribution requirements (CI-H, HASS, REST, LAB), prerequisites, scheduling, course comparisons, and degree-specific electives.

## Architecture

**Data** — Course data is fetched from [Hydrant](https://hydrant.mit.edu) (Spring 2026 + Fall 2025), which provides clean, structured JSON with descriptions, prereqs, instructors, schedules, and distribution tags. Run `python -m src.scraper` to refresh `data/courses.json`.

**Retrieval** — `src/retriever.py` builds a BM25 index over all ~3000 courses. BM25 handles document length normalisation and term frequency saturation, which matters for matching short student queries against variable-length course descriptions.

**Three-pass RAG pipeline** (`src/chat.py`):
1. **Profile** — A lightweight LLM pass (`extract_profile_updates`) reads each message and updates a persistent `StudentProfile` (year, major, completed/in-progress courses, remaining requirements, constraints, interests, program context).
2. **Preflight** — A planning LLM call uses the profile to output a JSON retrieval plan (web searches, catalog queries, level filter), skipping already-known info and tailoring queries to outstanding requirements and constraints.
3. **Gather** — Executes the plan: runs web searches for unfamiliar MIT programs (e.g. TPP, UROP, CRE), runs multiple BM25 queries against the catalog with filters/boosts driven by the profile (excluding completed/in-progress courses, boosting remaining requirements, respecting semester hints), and exact-matches any course numbers mentioned explicitly.
4. **Answer** — A final LLM call answers using the profile, catalog excerpt, web context, and evaluation data, applying the student’s stated schedule/workload preferences when ranking and commenting on suggestions.

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
