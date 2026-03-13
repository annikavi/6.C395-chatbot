"""
MIT Course Catalog Chatbot — improved with student profile tracking and
evaluation data enrichment.

Architecture (three-pass RAG):
  1. PROFILE — extract student info from message, update persistent profile
  2. PREFLIGHT — LLM identifies knowledge gaps → structured retrieval plan
  3. GATHER — web searches + BM25 retrieval (excluding taken courses,
               boosting outstanding requirements)
  4. ANSWER — LLM answers with full context: profile + web + catalog + evals
"""

from __future__ import annotations

import json
import re
from typing import Optional

from huggingface_hub import InferenceClient

from config import BASE_MODEL, MY_MODEL, HF_TOKEN
from src.retriever import CourseRetriever
from src.web_search import web_search
from src.profile import StudentProfile, extract_profile_updates


# ── Preflight prompt ──────────────────────────────────────────────────────────

_PREFLIGHT_PROMPT_TEMPLATE = """\
You are a planning assistant for an MIT course advisor chatbot. Your ONLY job
is to analyse the student's message and output a JSON retrieval plan.

You have access to the student's profile (below). Use it to:
- Infer the right level_filter from their year/major
- Generate catalog_queries that account for what they've already taken
- Skip web searches for things already known from their profile

STUDENT PROFILE:
{profile_block}

Output ONLY valid JSON with these fields:
{{
  "web_searches": ["query1", "query2"],
  "catalog_queries": ["query1", "query2"],
  "level_filter": "graduate" | "undergraduate" | null
}}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 1 — web_searches (max 2, often [])
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ONLY for MIT institutional programs/offices NOT in the course catalog:
  • Named MIT graduate/special programs: TPP, LGO, SDM, MBAn, HST, MISTI, UROP, UPOP
  • Program-specific curriculum terms: CRE, thesis requirement, named concentration
  • Named MIT office/policy: Communication Requirement office, SHASS requirement

DO NOT search for: subject areas, Course 6-X tracks, distribution tags (CI-H etc.),
GIRs, general scheduling questions, or anything resolvable from the catalog.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 2 — catalog_queries (1–4 queries)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
BM25 keyword strings. MIT dept → number prefix:
  Math=18, Physics=8, CS/EE=6, Biology=7, Chemistry=5, Economics=14,
  Management=15, Political Science=17, History=21H, Writing=21W,
  Architecture=4, Urban Studies=11, Earth Science=12, Aero/Astro=16,
  Neuro/Cog Sci=9, Nuclear=22, Philosophy/Linguistics=24, Media Lab=MAS

If the student has outstanding requirements (from their profile), include
those tags in your queries to retrieve matching courses.
Make queries DIVERSE — different angles of the same request.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
RULE 3 — level_filter
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"undergraduate" for: freshman/sophomore/junior/senior, SB degree, undergrad context
"graduate"      for: MEng, SM, PhD, TPP, LGO, SDM, MBAn, grad student
null            for: unclear or mixed

Output ONLY the JSON object, no explanation, no markdown fences.
"""

# Backwards-compatible alias for any remaining references
_PREFLIGHT_PROMPT = _PREFLIGHT_PROMPT_TEMPLATE


# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT_TEMPLATE = """\
You are an expert MIT academic advisor specialising in course selection.
You help MIT students find courses that fit their major, year, requirements,
schedule, interests, and workload preferences.

## STUDENT PROFILE
(Updated throughout this conversation — always use this as context.)

{student_profile_block}

## Your capabilities
- Recommend courses matching specific distribution requirements (CI-H, CI-M, HASS-A/H/S, REST, LAB)
- Explain prerequisites and how courses build on each other
- Reason about scheduling conflicts and realistic course loads
- Compare courses using real student evaluation data (ratings, hours/week)
- Warn when a recommended course conflicts with completed/in-progress courses
- Proactively notice profile gaps and ask for them naturally

## CRITICAL ground rules
1. Only recommend courses from the COURSE CATALOG EXCERPT below.
   Do NOT invent courses or cite numbers not in the excerpt.
2. NEVER recommend courses the student has already completed or is currently taking
   (check the profile's completed and in_progress lists).
3. Always flag prerequisites the student may not have met based on their completed courses.
4. When evaluation data is shown (⭐ rating, ⏱ hours), use it to give nuanced advice
   — e.g. warn about high-workload courses, highlight highly-rated instructors.
5. When a student has multiple constraints (e.g. "CI-H about AI ethics"), find courses
   satisfying ALL constraints together. Do NOT split into separate lists.
6. When building a schedule, parse schedule/time fields and verify ZERO time overlaps.
   Two courses conflict if they share any day AND their time ranges overlap.
7. Be concise but complete — students are busy. Use bullet points for lists.
8. If uncertain, say so rather than guessing. Direct to student.mit.edu/catalog for
   anything outside the excerpt. 
9. If the student mentions they just finished a requirement, acknowledge it and
   update your recommendation focus accordingly.
10. Proactively notice if the profile suggests a requirement is almost completable
    with one more course, and mention it.

## Anti-repetition and response structure (VERY IMPORTANT)
- Never repeat the same set of recommendations multiple times.
- Do not restate a list under multiple headings (e.g., "Considering X..." then "However, considering Y..." with the same courses).
- Provide ONE consolidated list of at most 6 courses, sorted by best fit.
- Each course should have a single, non-redundant rationale line tied to the student's constraints.
- If you need additional info to answer (e.g., ambiguous acronyms), ask 1–2 short clarifying questions instead of guessing.

## Handling "CRE" questions (Course Restricted Elective)
- "CRE" is program-specific. If the student's program/track and allowed CRE list is not in WEB CONTEXT, you MUST ask:
  1) Which program (e.g., TPP / IDS / SDM / LGO / other) and which track, if any?
  2) If they have a link or excerpt of the program's CRE list they can paste.
- Do NOT assume the student's program from a generic major selection.

## Evaluation data interpretation
When a course has evaluation data, interpret it as:
  ⭐ Overall rating: X/7 — student satisfaction (6+ is excellent, 5+ is good)
  ⏱ Hours/week: X outside class — workload indicator (4-6 is typical, 8+ is heavy)
  👨‍🏫 Instructor rating: X/7 — teaching quality
  📊 Response rate: X% — reliability (below 30% means fewer responses, interpret cautiously)

## MIT Schedule Notation
Day codes: M=Mon, T=Tue, W=Wed, R=Thu, F=Fri
Example: "Lec: TR 1-2:30 (34-101)" = lecture Tue+Thu 1:00-2:30pm in room 34-101
When presenting a schedule, list each course ONCE with days and times.

## MIT Degree Requirements Reference
**GIRs**: Science Core (18.01, 18.02, 8.01, 8.02, Chemistry, Biology), REST (2), HASS (8), LAB (1), PE (4)

**HASS breakdown** (8 subjects total):
  • 1 HASS-A (Arts), 1 HASS-H (Humanities), 1 HASS-S (Social Sciences)
  • 2 HASS Electives, 1 CI-H, 1 CI-M (in major)
  Note: one course can satisfy multiple tags (e.g. CI-H + HASS-H counts for both)

**Course 6 Tracks**:
  6-1 (EE): 6.1910, 6.2000, 6.2050, 6.3000
  6-2 (EE+CS): 6.1010, 6.1910, 6.2000, 6.3000, 6.3900; CI-M via 6.1800
  6-3 (CS): 6.1010→6.1020→6.1040; 6.1200J; 6.1210; 6.1800 (CI-M); 6.3700; 6.3900
  6-4 (AI+DM): 6.1010, 6.1200J, 6.1210, 6.3700, 6.3900, 6.3800, 6.4100/6.4110
  6-7 (CS+Bio): 6-3 core + biology
  6-9 (CS+Cognition): CS core + Course 9

## WEB CONTEXT
{web_context_block}

## COURSE CATALOG EXCERPT
{catalog_excerpt_block}

---
Always cross-reference the student's profile before making recommendations.
Only cite courses from the catalog excerpt; for anything else direct to student.mit.edu/catalog.
"""

_TOP_K = 12


# ── Course formatting ─────────────────────────────────────────────────────────

def _format_eval_line(eval_data: dict) -> str:
    """Format evaluation data into a compact summary line."""
    if not eval_data:
        return ""

    parts = []
    overall = eval_data.get("overall_rating")
    if overall is not None:
        rr = eval_data.get("response_rate_pct", 0)
        reliability = f" ({rr:.0f}% response rate)" if rr and rr < 40 else ""
        parts.append(f"⭐ {overall:.1f}/7{reliability}")

    instr = eval_data.get("avg_instructor_rating")
    if instr is not None:
        parts.append(f"👨‍🏫 {instr:.1f}/7")

    hours = eval_data.get("avg_hours_outside_class")
    if hours is not None:
        workload_tag = ""
        if hours >= 8:
            workload_tag = " ⚠️ heavy"
        elif hours <= 4:
            workload_tag = " ✓ light"
        parts.append(f"⏱ {hours:.1f}h/wk outside class{workload_tag}")

    hrs_in = eval_data.get("avg_hours_in_class")
    if hrs_in is not None:
        parts.append(f"({hrs_in:.1f}h in class)")

    term = eval_data.get("term", "")
    term_label = {"2026FA": "Fall '25", "2025SP": "Spring '25"}.get(term, term)
    if term_label:
        parts.append(f"[{term_label} data]")

    return "  " + " | ".join(parts) if parts else ""


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

    # Evaluation data
    eval_line = _format_eval_line(course.get("eval", {}))
    if eval_line:
        lines.append(eval_line)

    # Optional catalog fields
    extras: list[str] = []
    if course.get("same_as"):
        extras.append(f"Same as: {course['same_as']}")
    if course.get("limited_enrollment"):
        extras.append("⚠️ Limited enrollment")
    if course.get("new_course"):
        extras.append("🆕 New course")
    if course.get("half_semester"):
        extras.append("Half-semester")
    if course.get("has_final"):
        extras.append("Has final exam")
    if extras:
        lines.append(f"  {' | '.join(extras)}")

    lines.append(f"  {course.get('description', '')[:300]}")
    return "\n".join(lines)


# ── History normalisation ──────────────────────────────────────────────────────

def _normalise_history(history: list) -> list[tuple[str, str]]:
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


# ── Chatbot ────────────────────────────────────────────────────────────────────

class Chatbot:
    """
    MIT Course Catalog chatbot using profile-aware three-pass RAG.

    Pass 0 (profile):   Extract student info from message, update profile.
    Pass 1 (preflight): LLM identifies knowledge gaps → retrieval plan.
    Pass 2 (gather):    Web searches + BM25 retrieval (profile-filtered).
    Pass 3 (answer):    LLM answers with profile + web + catalog + evals.

    Usage:
        chatbot = Chatbot()
        response, profile_summary = chatbot.get_response("I need a CI-H", history=[])
    """

    def __init__(self) -> None:
        model_id = MY_MODEL if MY_MODEL else BASE_MODEL
        self.model_id = model_id
        self.client = InferenceClient(model=model_id, token=HF_TOKEN)
        self.retriever = CourseRetriever()
        self.profile = StudentProfile()
        print(f"[Chatbot] Loaded {self.retriever.total_courses} courses.")

    def reset_profile(self) -> None:
        """Clear the student profile (e.g. when a new conversation starts)."""
        self.profile = StudentProfile()

    def apply_structured_profile(
        self,
        year: Optional[str] = None,
        major: Optional[str] = None,
        semester: Optional[str] = None,
        completed_csv: Optional[str] = None,
        requirements_remaining: Optional[list[str]] = None,
        constraints_text: Optional[str] = None,
    ) -> None:
        """
        Merge structured profile inputs from the UI into the StudentProfile.

        This bypasses LLM extraction for basic fields like year/major/courses
        and lets the user set them via dropdowns/text boxes.
        """
        updates: dict = {}

        if year:
            y = year.strip().lower()
            year_map = {
                "first-year": "freshman",
                "first year": "freshman",
                "freshman": "freshman",
                "sophomore": "sophomore",
                "junior": "junior",
                "senior": "senior",
                "graduate": "graduate",
                "grad": "graduate",
            }
            updates["year"] = year_map.get(y, None)

        if major:
            updates["major"] = major.strip()

        if semester:
            s = semester.strip().lower()
            if s in ("fall", "spring"):
                updates["semester"] = s

        if completed_csv:
            courses = [
                c.strip()
                for c in completed_csv.split(",")
                if c.strip()
            ]
            if courses:
                updates["completed"] = courses

        if requirements_remaining:
            updates["requirements_remaining"] = requirements_remaining

        if constraints_text:
            constraints = [c.strip() for c in constraints_text.split(";") if c.strip()] or [
                constraints_text.strip()
            ]
            updates["constraints"] = constraints

        if updates:
            self.profile.merge(updates)

    def set_structured_profile(
        self,
        year: Optional[str] = None,
        major: Optional[str] = None,
        semester: Optional[str] = None,
        completed_csv: Optional[str] = None,
        requirements_remaining: Optional[list[str]] = None,
        constraints_text: Optional[str] = None,
    ) -> None:
        """
        Set (overwrite) structured profile fields from the UI.

        Unlike apply_structured_profile(), this is intended for live UI updates where
        the user may remove/replace items. It overwrites only the fields provided.
        """
        year_map = {
            "first-year": "freshman",
            "first year": "freshman",
            "freshman": "freshman",
            "sophomore": "sophomore",
            "junior": "junior",
            "senior": "senior",
            "graduate": "graduate",
            "grad": "graduate",
        }

        if year is not None:
            y = year.strip().lower()
            self.profile.year = year_map.get(y, None)

        if major is not None:
            m = major.strip()
            self.profile.major = m or None

        if semester is not None:
            s = semester.strip().lower()
            self.profile.semester = s if s in ("fall", "spring") else None

        if completed_csv is not None:
            completed = [
                c.strip().upper()
                for c in completed_csv.split(",")
                if c.strip()
            ]
            self.profile.completed = list(dict.fromkeys(completed))
            # Ensure in_progress doesn't contain any completed courses
            self.profile.in_progress = [
                c for c in self.profile.in_progress if c not in set(self.profile.completed)
            ]

        if requirements_remaining is not None:
            reqs = [r.strip().upper() for r in requirements_remaining if r and r.strip()]
            self.profile.requirements_remaining = list(dict.fromkeys(reqs))

        if constraints_text is not None:
            raw = constraints_text.strip()
            if not raw:
                self.profile.constraints = []
            else:
                constraints = [c.strip() for c in raw.split(";") if c.strip()]
                self.profile.constraints = list(dict.fromkeys(constraints or [raw]))

    # ── Pass 0: Profile extraction ─────────────────────────────────────────────

    def _update_profile(self, message: str) -> None:
        updates = extract_profile_updates(message, self.client, self.model_id)
        if updates:
            self.profile.merge(updates)

    # ── Pass 1: Preflight ──────────────────────────────────────────────────────

    def _run_preflight(self, message: str, history: list) -> dict:
        profile_block = self.profile.to_prompt_block()
        preflight_system = _PREFLIGHT_PROMPT_TEMPLATE.replace(
            "{profile_block}", profile_block
        )
        preflight_messages: list[dict] = [
            {
                "role": "system",
                "content": preflight_system,
            }
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
                max_tokens=350,
                temperature=0.0,
            )
            raw = resp.choices[0].message.content.strip()
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
        history: list,
        catalog_excerpt: str,
        web_context: str = "",
    ) -> list[dict]:
        # If the user asks about a program-specific requirement (e.g., CRE) and we
        # don't have program context yet, nudge the model to ask clarifying questions
        # rather than guessing.
        if re.search(r"\bcre\b", message, flags=re.IGNORECASE) and not (
            self.profile.program_context or self.profile.major
        ):
            message = (
                message
                + "\n\n(If CRE depends on my specific program/track, ask me what program I'm in "
                + "and what CRE list/rules I should follow before recommending courses.)"
            )
        system_content = (
            _SYSTEM_PROMPT_TEMPLATE
            .replace("{student_profile_block}", self.profile.to_prompt_block())
            .replace("{catalog_excerpt_block}", catalog_excerpt)
            .replace("{web_context_block}", web_context or "None")
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

    def get_response(
        self,
        message: str,
        history: list,
    ) -> tuple[str, str]:
        """
        Generate a response using the full pipeline.

        Returns:
            (response_text, profile_summary_for_display)
        """
        if not message.strip():
            return (
                "Hi! I'm your MIT course advisor. Tell me about yourself — "
                "your year, major, what you've already taken — and I'll help "
                "you find the right courses.",
                self.profile.to_prompt_block(),
            )

        # ── Pass 0: Update profile ────────────────────────────────────────────
        self._update_profile(message)

        # ── Pass 1: Preflight ─────────────────────────────────────────────────
        plan = self._run_preflight(message, history)
        web_queries    = plan.get("web_searches", [])[:2]
        catalog_queries = plan.get("catalog_queries", [message])[:4]
        level_filter   = plan.get("level_filter")

        # Infer level from profile if preflight didn't set it
        if not level_filter and self.profile.year == "graduate":
            level_filter = "graduate"
        elif not level_filter and self.profile.year in ("freshman", "sophomore", "junior", "senior"):
            level_filter = "undergraduate"
        elif not level_filter and self.profile.major and any(
            kw in (self.profile.major or "").upper()
            for kw in ("MENG", "SM", "PHD", "TPP", "LGO", "SDM", "MBAN", "HST")
        ):
            level_filter = "graduate"

        # ── Gather: Web searches ──────────────────────────────────────────────
        web_lines: list[str] = []
        for query in web_queries:
            snippet = web_search(query)
            if snippet:
                web_lines.append(f"**{query}**: {snippet}")
                print(f"[WebSearch] '{query}' → {len(snippet)} chars")
        web_context = "\n\n".join(web_lines)

        # ── Gather: Course retrieval ──────────────────────────────────────────
        exclude = set(self.profile.completed) | set(self.profile.in_progress)
        boost   = self.profile.requirements_remaining

        seen_numbers: set[str] = set()
        retrieved: list[dict] = []

        for query in catalog_queries:
            results = self.retriever.retrieve(
                query,
                top_k=_TOP_K,
                level_filter=level_filter,
                exclude_numbers=list(exclude),
                boost_distributions=boost if boost else None,
                offered_hint=self.profile.semester,
            )
            for course in results:
                num = course["number"]
                if num not in seen_numbers:
                    seen_numbers.add(num)
                    retrieved.append(course)

        # Exact-match lookup for explicitly mentioned course numbers
        mentioned = re.findall(r"\b\d+\.\w+\b", message)
        for num in mentioned:
            if num not in seen_numbers:
                exact = self.retriever.get_by_number(num)
                if exact:
                    seen_numbers.add(num)
                    retrieved.insert(0, exact)

        # De-duplicate cross-listed / meets-with courses so we don't suggest
        # the same class twice under different numbers (e.g., 6.5220 and 18.416).
        def _aliases(course: dict) -> set[str]:
            out = {str(course.get("number", "")).upper()}
            for key in ("same_as", "meets_with"):
                raw = course.get(key) or ""
                if isinstance(raw, str) and raw.strip():
                    parts = re.split(r"[,\s]+", raw.strip())
                    for p in parts:
                        if p:
                            out.add(p.upper())
            return out

        seen_aliases: set[str] = set()
        deduped: list[dict] = []
        for c in retrieved:
            aliases = _aliases(c)
            if aliases & seen_aliases:
                continue
            seen_aliases |= aliases
            deduped.append(c)

        retrieved_top = deduped[:_TOP_K * 2]

        # ── Pass 2: Format and answer ─────────────────────────────────────────
        if retrieved_top:
            catalog_excerpt = "\n\n".join(_format_course(c) for c in retrieved_top)
        else:
            catalog_excerpt = "No courses found matching your query. Please check student.mit.edu/catalog."

        messages = self._build_messages(message, history, catalog_excerpt, web_context)

        try:
            response = self.client.chat_completion(
                messages=messages,
                max_tokens=1200,
                temperature=0.35,
                top_p=0.9,
            )
            answer = response.choices[0].message.content.strip()
            return answer, self.profile.to_prompt_block()

        except Exception as e:
            error_str = str(e)
            if "503" in error_str:
                msg = "The model is currently busy (503). Please wait a moment and try again."
            elif "402" in error_str:
                msg = "Free-tier inference credits are temporarily exhausted (402). Try again later."
            else:
                msg = f"An error occurred: {error_str}"
            return msg, self.profile.to_prompt_block()