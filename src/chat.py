"""
MIT Course Catalog Chatbot

RAG pipeline (two-pass architecture):
  1. PREFLIGHT — an LLM pass that reads the student's message, identifies
     knowledge gaps (unknown terms, programs, requirements), and outputs
     structured search/retrieval instructions.
  2. GATHER — execute the preflight instructions: web searches for MIT
     context + multiple BM25 retrieval queries across the course catalog.
  3. ANSWER — a second LLM pass that reasons over the gathered context to
     produce the final response.

This lets the LLM *itself* decide what it needs to look up rather than
relying on static keyword matching.
"""

from __future__ import annotations

import json
import re

from huggingface_hub import InferenceClient

from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from src.retriever import CourseRetriever
from src.web_search import web_search


# ── Preflight prompt ──────────────────────────────────────────────────────────
# Pass 1: the LLM reads the student's question and tells us what to look up.

_PREFLIGHT_PROMPT = """\
You are a planning assistant for an MIT course advisor chatbot. Your ONLY job \
is to analyse the student's message and output a JSON retrieval plan.

You must output ONLY valid JSON with these fields:
{
  "web_searches": ["query1", "query2"],
  "catalog_queries": ["query1", "query2"],
  "level_filter": "graduate" | "undergraduate" | null
}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 — web_searches (max 2, often empty [])
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Web search is ONLY for MIT institutional programs/offices/policies that are
NOT themselves courses — things like degree programs, fellowship programs,
administrative terms, or curriculum requirements specific to a named MIT program.

SEARCH when the student mentions (written out OR abbreviated):
  • A named MIT graduate or special program they belong to:
    e.g. "Technology and Policy Program", "TPP", "Leaders for Global Operations", "LGO",
    "System Design and Management", "SDM", "Master in Business Analytics", "MBAn",
    "Health Sciences and Technology", "HST", "MISTI", "UROP", "UPOP"
  • A program-specific curriculum term:
    e.g. "CRE" (Course Restricted Elective), "thesis requirement", "concentration requirement",
    "departmental program", a named concentration within a degree
  • A named MIT office/policy that has requirements:
    e.g. "Communication Requirement office", "SHASS requirement"

DO NOT SEARCH for:
  • Academic subject areas — these are in the course catalog:
    "machine learning", "robotics", "computational social science", "technology policy",
    "economics", "philosophy", "neuroscience", "quantum computing", etc.
  • Standard MIT degree programs by number: "Course 6-3", "Course 2", "Course 18" —
    you already know these from the system prompt
  • Distribution tags: CI-H, CI-M, HASS-A/H/S, REST, LAB — already defined
  • GIRs (Science Core, HASS, LAB, REST, PE) — already defined
  • Words like "prereqs", "dept", "conc" — just standard abbreviations
  • General questions about picking classes, scheduling, workload

EXAMPLE: "I'm a Course 6-3 junior, help me pick classes" → web_searches: []
EXAMPLE: "I'm a TPP student, what CRE should I take?" → web_searches: ["MIT Technology and Policy Program curriculum", "MIT TPP course restricted elective CRE requirements"]
EXAMPLE: "what's a good CI-H about AI ethics?" → web_searches: []

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 2 — catalog_queries (1–3 queries)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BM25 keyword strings to retrieve relevant courses. Think about what words
would appear in course titles and descriptions.

MIT department → course number prefix mapping:
  Math=18, Physics=8, CS/EE=6, Biology=7, Chemistry=5, Economics=14,
  Management/Sloan=15, Political Science=17, History=21H, Writing=21W,
  Architecture=4, Urban Studies=11, Earth Science=12, Aero/Astro=16,
  Neuro/Cog Sci=9, Nuclear=22, Philosophy/Linguistics=24, Media Lab=MAS

Tips:
  • If the student asks for a SUBJECT, include the dept prefix (e.g. "18. linear algebra")
  • Make queries DIVERSE to cover different angles of the request
  • If student wants DISTRIBUTION + TOPIC, combine both: "CI-H ethics technology policy"
  • Do NOT just echo the student's words — think about what course descriptions say

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 3 — level_filter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  "undergraduate" if: Course 6-1/2/3/4/7/9, SB degree, undergrad context
  "graduate"     if: TPP, LGO, SDM, MBAn, MEng, SM, PhD, grad student context
  null           if: unclear or mixed

Output ONLY the JSON object, no explanation, no markdown fences.
"""


# ── System prompt template ─────────────────────────────────────────────────────
_SYSTEM_PROMPT = """\
You are an expert MIT academic advisor specialising in course selection. \
You help MIT students navigate the course catalog to find courses that fit \
their major, year, distribution requirements, schedule, and interests.

## Your capabilities
- Recommend courses matching specific distribution requirements (CI-H, CI-M, HASS-A/H/S, REST, LAB)
- Explain prerequisites and how to satisfy them
- Reason about scheduling conflicts and course load
- Compare courses across departments
- Track what a student has told you about their situation across the conversation

## Ground rules — CRITICAL
1. Only recommend courses from the COURSE CATALOG EXCERPT provided to you below.
   Do NOT invent courses or cite course numbers not in the excerpt.
2. If a student asks about a course not in the excerpt, say you don't have full \
   details and suggest they check student.mit.edu/catalog for the official listing.
3. Always flag prerequisites the student may not have met.
4. When mentioning schedules or instructor names, note that these may change and \
   recommend confirming at student.mit.edu/catalog.
5. Be concise but complete — students are busy.
6. If you're uncertain, say so explicitly rather than guessing.
8. When a student has MULTIPLE constraints (e.g. "CI-H about AI ethics", "HASS-S \
   that covers economics"), find courses that satisfy ALL constraints together. \
   Do NOT split into separate lists. A CI-H about AI ethics means: find courses \
   that are tagged CI-H AND whose topic relates to AI ethics. Check the \
   distributions field AND the description/title of each course in the excerpt.
7. When building a schedule or recommending multiple courses together, you MUST \
   parse the schedule/time fields and verify there are ZERO time overlaps. \
   Two courses conflict if they share any day AND their time ranges overlap. \
   For each pair of courses you recommend together, explicitly check for conflicts \
   before including them. If the student's preferences (e.g. "afternoon only") \
   make a conflict-free schedule impossible, say so and offer alternatives.

## MIT Schedule Notation
Day codes: M=Monday, T=Tuesday, W=Wednesday, R=Thursday, F=Friday.
Combined codes: MW=Mon+Wed, TR=Tue+Thu, MWF=Mon+Wed+Fri, MTWRF=every weekday.
Example: "Lec: TR 1-2:30 (34-101)" means lecture on Tuesday AND Thursday, 1:00-2:30pm, room 34-101.
When presenting a schedule, list each course ONCE with its days and times exactly as shown \
in the schedule field. Do NOT split a single course across separate day entries. \
Format example:
  - [18.06] Linear Algebra — MW 11:00-12:00
  - [6.1010] Fundamentals of Programming — MWF 10:00-11:00

## MIT Department Numbering
When a student asks for a subject by name (e.g. "math course"), match it to the \
correct MIT department number. Key mappings:
- Math = Course 18 (e.g. 18.01, 18.06)    - Physics = Course 8
- CS = Course 6                             - Biology = Course 7
- Chemistry = Course 5                      - Economics = Course 14
- Management = Course 15 (Sloan)            - Political Science = Course 17
If the student says "math course", look for courses starting with "18.". \
If they say something interdisciplinary like "computational social science", \
search by topic keywords across all departments.

## MIT Degree Requirements Reference
**GIRs (General Institute Requirements)**
- Science Core: 18.01, 18.02, 8.01, 8.02, Chemistry (5.111/5.112), Biology (7.01x)
- REST (2 subjects): courses tagged REST; at least one in the student's major area
- HASS (8 subjects total):
  • 1 HASS-A (Arts)
  • 1 HASS-H (Humanities)
  • 1 HASS-S (Social Sciences)
  • 2 HASS Electives (any HASS subject)
  • 1 CI-H (Communication Intensive — writing-intensive HASS subject)
  • 1 CI-M (Communication Intensive in the Major — embedded in degree program)
  Note: One course can satisfy multiple tags (e.g., a CI-H + HASS-H counts for both)
- LAB (1 subject): Institute Lab requirement
- PE (4 semesters)

**Course 6 Tracks**
- 6-1 (EE): 6.1910, 6.2000, 6.2050 (LAB), 6.3000, plus EE program subjects
- 6-2 (EE+CS): 6.1010, 6.1910, 6.2000, 6.3000, plus 6.3900 or similar; CI-M via 6.1800
- 6-3 (CS): 6.1010→6.1020→6.1040; 6.1200J; 6.1210; 6.1800 (CI-M); 6.3700; 6.3900; plus 48 units electives
- 6-4 (AI+DM): 6.1010, 6.1200J, 6.1210, 6.3700, 6.3900, 6.3800, 6.4100 or 6.4110; plus AI electives
- 6-7 (CS+Bio): 6-3 core + biology subjects
- 6-9 (Computation+Cognition): CS core + brain/cognitive science subjects (joint with Course 9)

## WEB CONTEXT
(Looked up to clarify MIT-specific terms, programs, or requirements mentioned in this conversation.)

{web_context}

## COURSE CATALOG EXCERPT
(Real courses retrieved from the MIT catalog based on this conversation.)

{catalog_excerpt}

---
Use the web context to understand any MIT-specific jargon, programs, or requirements the student mentioned.
Only cite courses from the catalog excerpt above; for anything else direct the student to student.mit.edu/catalog.
"""

# How many courses to inject per turn
_TOP_K = 12


def _format_course(course: dict) -> str:
    """Format a single course dict into a compact, readable string."""
    dists = ", ".join(course.get("distributions", [])) or "None"
    lines = [
        f"**[{course['number']}] {course['title']}**",
        f"  Units: {course.get('units', 'N/A')} | Level: {course.get('level', 'N/A')} | Distributions: {dists}",
        f"  Prereqs: {course.get('prereqs', 'None')}",
        f"  Offered: {course.get('offered', 'See catalog')} | Schedule: {course.get('schedule', 'See catalog')}",
        f"  Instructors: {course.get('instructors', 'See catalog')}",
    ]
    # Optional enrichment fields
    extras: list[str] = []
    if "rating" in course:
        extras.append(f"Rating: {course['rating']}/7")
    if "avg_hours" in course:
        extras.append(f"Avg hours/wk: {course['avg_hours']}")
    if "avg_class_size" in course:
        extras.append(f"Class size: ~{int(course['avg_class_size'])}")
    if course.get("same_as"):
        extras.append(f"Same as: {course['same_as']}")
    if course.get("limited_enrollment"):
        extras.append("Limited enrollment")
    if course.get("new_course"):
        extras.append("NEW course")
    if course.get("half_semester"):
        extras.append("Half-semester")
    if course.get("has_final"):
        extras.append("Has final exam")
    if extras:
        lines.append(f"  {' | '.join(extras)}")
    lines.append(f"  {course.get('description', '')}")
    return "\n".join(lines)


def _normalise_history(history: list) -> list[tuple[str, str]]:
    """
    Normalise Gradio history to a list of (user, assistant) string pairs.

    Gradio ≥4.x passes history as a list of {"role": ..., "content": ...} dicts.
    Older versions pass [[user_msg, assistant_msg], ...] pairs.
    """
    if not history:
        return []
    if isinstance(history[0], dict):
        pairs: list[tuple[str, str]] = []
        user_buf = ""
        for msg in history:
            role = msg.get("role", "")
            content = msg.get("content", "") or ""
            if role == "user":
                user_buf = content
            elif role == "assistant":
                pairs.append((user_buf, content))
                user_buf = ""
        return pairs
    return [(u or "", a or "") for u, a in history]



class Chatbot:
    """
    MIT Course Catalog chatbot using two-pass RAG.

    Pass 1 (preflight): LLM identifies knowledge gaps → search queries.
    Pass 2 (answer):    LLM answers with gathered context.

    Usage:
        chatbot = Chatbot()
        response = chatbot.get_response("I need a CI-H for my 6-3 degree", history=[])
    """

    def __init__(self) -> None:
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.retriever = CourseRetriever()
        print(f"[Chatbot] Loaded {self.retriever.total_courses} courses from catalog.")

    # ── Pass 1: Preflight ──────────────────────────────────────────────────────

    def _run_preflight(self, message: str, history: list) -> dict:
        """
        Ask the LLM what it needs to look up before answering.

        Returns dict with keys: web_searches, catalog_queries, level_filter.
        Falls back to a simple default if the LLM output is malformed.
        """
        # Give the preflight LLM the last 2 turns for context
        preflight_messages: list[dict] = [
            {"role": "system", "content": _PREFLIGHT_PROMPT},
        ]
        for user_msg, assistant_msg in _normalise_history(history)[-2:]:
            if user_msg:
                preflight_messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                preflight_messages.append({"role": "assistant", "content": assistant_msg})
        preflight_messages.append({"role": "user", "content": message})

        try:
            resp = self.client.chat_completion(
                messages=preflight_messages,
                max_tokens=300,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if raw.startswith("```"):
                raw = re.sub(r"^```\w*\n?", "", raw)
                raw = re.sub(r"\n?```$", "", raw)
            plan = json.loads(raw)
            print(f"[Preflight] Plan: {json.dumps(plan, indent=2)}")
            return plan
        except Exception as e:
            print(f"[Preflight] Failed ({e}), using fallback")
            return {
                "web_searches": [],
                "catalog_queries": [message],
                "level_filter": None,
            }

    # ── Prompt construction ────────────────────────────────────────────────────

    def _build_messages(
        self,
        message: str,
        history: list[list[str]],
        catalog_excerpt: str,
        web_context: str = "",
    ) -> list[dict]:
        """Construct the full messages list for the chat completion API."""
        system_content = _SYSTEM_PROMPT.format(
            catalog_excerpt=catalog_excerpt,
            web_context=web_context or "None",
        )
        messages: list[dict] = [{"role": "system", "content": system_content}]

        for user_msg, assistant_msg in _normalise_history(history):
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

        messages.append({"role": "user", "content": message})
        return messages

    # ── Main response method ───────────────────────────────────────────────────

    def get_response(self, message: str, history: list[list[str]]) -> str:
        """
        Generate a response using two-pass architecture:
        1. Preflight: LLM decides what to search for
        2. Gather: run web searches + BM25 retrieval
        3. Answer: LLM responds with full context
        """
        if not message.strip():
            return "Please ask me something — I'm here to help with MIT course selection!"

        # ── Pass 1: Preflight ─────────────────────────────────────────────────
        plan = self._run_preflight(message, history)
        web_queries = plan.get("web_searches", [])[:3]
        catalog_queries = plan.get("catalog_queries", [message])[:3]
        level_filter = plan.get("level_filter")

        # ── Gather: Web searches ──────────────────────────────────────────────
        web_lines: list[str] = []
        for query in web_queries:
            snippet = web_search(query)
            if snippet:
                web_lines.append(f"**{query}**: {snippet}")
                print(f"[WebSearch] '{query}' → {len(snippet)} chars")

        web_context = "\n\n".join(web_lines)

        # ── Gather: Course retrieval (multiple diverse queries) ───────────────
        seen_numbers: set[str] = set()
        retrieved: list[dict] = []

        for query in catalog_queries:
            results = self.retriever.retrieve(
                query,
                top_k=_TOP_K,
                level_filter=level_filter,
            )
            for course in results:
                num = course["number"]
                if num not in seen_numbers:
                    seen_numbers.add(num)
                    retrieved.append(course)

        # Exact-match lookup for any course numbers mentioned explicitly
        mentioned = re.findall(r"\b\d+\.\w+\b", message)
        for num in mentioned:
            if num not in seen_numbers:
                exact = self.retriever.get_by_number(num)
                if exact:
                    seen_numbers.add(num)
                    retrieved.insert(0, exact)

        # Cap total courses to avoid blowing up context
        retrieved = retrieved[:_TOP_K * 2]

        # ── Pass 2: Format and answer ─────────────────────────────────────────
        if retrieved:
            catalog_excerpt = "\n\n".join(_format_course(c) for c in retrieved)
        else:
            catalog_excerpt = "No courses found matching your query. Please check student.mit.edu/catalog."

        messages = self._build_messages(message, history, catalog_excerpt, web_context)

        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=1024,
                temperature=0.4,
                top_p=0.9,
            )
            return response.choices[0].message.content.strip()

        except Exception as e:
            error_str = str(e)
            if "503" in error_str:
                return (
                    "The model is currently busy (503). "
                    "Please wait a few seconds and try again."
                )
            if "402" in error_str:
                return (
                    "Free-tier inference credits are temporarily exhausted (402). "
                    "Please try again later or check your HuggingFace account."
                )
            return f"An error occurred: {error_str}"
