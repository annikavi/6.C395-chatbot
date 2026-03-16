"""
BM25-based retrieval over the MIT course catalog (data/courses.json),
enriched with evaluation data from data/evaluations.json when available.

BM25 normalises for document length and saturates term frequency, giving
better precision than raw TF-IDF for short free-form student queries.

Evaluation enrichment:
  - Merges overall_subject_rating, instructor ratings, hours/week, and
    response_rate from the scraped OSE evaluation data into each course dict.
  - Falls back gracefully when evaluations.json is missing or a course has
    no evaluation data.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


_DATA_PATH  = Path(__file__).parent.parent / "data" / "courses.json"
_EVAL_PATH  = Path(__file__).parent.parent / "data" / "evaluations.json"

_K1 = 1.5   # term-frequency saturation
_B  = 0.75  # length normalisation strength

_DIST_ALIASES: dict[str, str] = {
    "ci-h": "CI-H", "cih": "CI-H", "communication intensive": "CI-H",
    "ci-m": "CI-M", "cim": "CI-M",
    "hass-a": "HASS-A", "hassa": "HASS-A", "arts": "HASS-A",
    "hass-h": "HASS-H", "hassh": "HASS-H", "humanities": "HASS-H",
    "hass-s": "HASS-S", "hasss": "HASS-S", "social science": "HASS-S",
    "hass": "HASS",
    "rest": "REST",
    "lab": "LAB", "institute lab": "LAB",
    "plab": "PLAB", "partial lab": "PLAB",
}


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text.split()


def _course_document(course: dict) -> str:
    fields = [
        course.get("number", ""),
        course.get("title", ""),
        course.get("description", ""),
        course.get("prereqs", ""),
        " ".join(course.get("distributions", [])),
        course.get("department", ""),
        course.get("instructors", ""),
        course.get("offered", ""),
    ]
    return " ".join(f for f in fields if f)


def _load_evaluation_index(eval_path: Path) -> dict[str, dict]:
    """
    Build a dict mapping canonical course number → evaluation summary.
    Aggregates across terms; uses the most recent term's data for each field.
    """
    if not eval_path.exists():
        print("[Retriever] No evaluations.json found — running without eval data.")
        return {}

    try:
        with eval_path.open() as f:
            raw = json.load(f)
    except Exception as e:
        print(f"[Retriever] Could not load evaluations.json: {e}")
        return {}

    index: dict[str, dict] = {}

    # Process terms in chronological order so newer data wins
    term_order = {"2025SP": 0, "2026FA": 1}
    terms_sorted = sorted(
        raw.get("terms", {}).items(),
        key=lambda kv: term_order.get(kv[0], -1)
    )

    for term_id, term_data in terms_sorted:
        for course in term_data.get("courses", []):
            if "error" in course:
                continue

            # A course page may list multiple subject numbers (e.g. 1.010B and 1.10)
            # Index under all of them
            numbers = course.get("subject_numbers", [])
            if not numbers:
                continue

            # Build eval summary
            overall = course.get("overall_subject_rating") or {}
            hours_out = course.get("hours_per_week_outside_class") or {}
            hours_in  = course.get("hours_per_week_in_class") or {}
            pace      = course.get("pace") or {}
            subj_ratings = course.get("subject_ratings", {}) or {}

            # Summarise instructor ratings: take average of unique instructors' overall ratings
            instr_overalls = []
            for instr in course.get("instructors", []):
                r = (instr.get("ratings") or {}).get("Overall rating")
                if isinstance(r, dict) and r.get("avg") is not None:
                    instr_overalls.append(r["avg"])
                elif isinstance(r, (int, float)):
                    instr_overalls.append(float(r))

            eval_entry = {
                "term": term_id,
                "response_rate_pct": course.get("response_rate_pct"),
                "total_respondents": course.get("total_respondents"),
                "overall_rating": overall.get("avg"),
                "overall_rating_median": overall.get("median"),
                "overall_rating_out_of": overall.get("out_of", 7),
                "avg_hours_outside_class": hours_out.get("avg"),
                "avg_hours_in_class": hours_in.get("avg"),
                "pace_avg": pace.get("avg"),  # 4=just right
                "avg_instructor_rating": (
                    round(sum(instr_overalls) / len(instr_overalls), 2)
                    if instr_overalls else None
                ),
                "instructors_detail": [
                    {
                        "name": i.get("name"),
                        "role": i.get("role"),
                        "section_type": i.get("section_type"),
                        "ratings": i.get("ratings"),
                    }
                    for i in course.get("instructors", [])
                ],
                "subject_ratings": subj_ratings,
            }

            for num in numbers:
                # Normalise: strip trailing letters like "B" in "1.010B"
                index[num.upper()] = eval_entry
                # Also index without suffix: "1.010B" → also stored as "1.010"
                base = re.sub(r"[A-Z]+$", "", num.upper())
                if base != num.upper() and base not in index:
                    index[base] = eval_entry

    print(f"[Retriever] Loaded evaluation data for {len(index)} course numbers.")
    return index


class CourseRetriever:
    """
    BM25 retrieval over the MIT course catalog, enriched with evaluation data.

    Usage:
        retriever = CourseRetriever()
        results = retriever.retrieve(
            "CI-H afternoon AI ethics junior 6-3",
            top_k=8,
            exclude_numbers=["6.1010", "18.01"],  # already taken
            boost_distributions=["CI-H"],          # prioritise requirement
        )
    """

    def __init__(self) -> None:
        self.courses: list[dict] = []
        self._eval_index: dict[str, dict] = {}
        self._idf: dict[str, float] = {}
        self._tf_norms: list[dict[str, float]] = []
        self._avg_doc_len: float = 0.0
        self._load_and_index()

    def _load_and_index(self) -> None:
        if not _DATA_PATH.exists():
            raise FileNotFoundError(
                f"Course data not found at {_DATA_PATH}. "
                "Run `python -m src.scraper` to generate data/courses.json."
            )
        with _DATA_PATH.open() as f:
            raw = json.load(f)
        self.courses = raw.get("courses", [])
        if not self.courses:
            raise ValueError("courses.json contains no courses.")

        # Load and merge evaluation data
        self._eval_index = _load_evaluation_index(_EVAL_PATH)
        self._enrich_courses_with_evals()

        self._build_bm25_index()
        print(f"[Retriever] Indexed {len(self.courses)} courses.")

    def _enrich_courses_with_evals(self) -> None:
        """Merge evaluation data into each course dict in-place."""
        enriched = 0
        for course in self.courses:
            num = course.get("number", "").upper()
            eval_data = self._eval_index.get(num)
            if not eval_data:
                # Try base number (strip trailing letters)
                base = re.sub(r"[A-Z]+$", "", num)
                eval_data = self._eval_index.get(base)
            if eval_data:
                course["eval"] = eval_data
                enriched += 1
        print(f"[Retriever] Enriched {enriched}/{len(self.courses)} courses with eval data.")

    def _build_bm25_index(self) -> None:
        N = len(self.courses)
        doc_tokens: list[list[str]] = []
        df: dict[str, int] = defaultdict(int)

        for course in self.courses:
            tokens = _tokenize(_course_document(course))
            doc_tokens.append(tokens)
            for term in set(tokens):
                df[term] += 1

        self._idf = {
            term: math.log((N - n + 0.5) / (n + 0.5) + 1)
            for term, n in df.items()
        }

        doc_lengths = [len(t) for t in doc_tokens]
        self._avg_doc_len = sum(doc_lengths) / N if N else 1.0

        self._tf_norms = []
        for tokens, doc_len in zip(doc_tokens, doc_lengths):
            tf_raw = Counter(tokens)
            norm = 1 - _B + _B * (doc_len / self._avg_doc_len)
            self._tf_norms.append({
                term: (count * (_K1 + 1)) / (count + _K1 * norm)
                for term, count in tf_raw.items()
            })

    def _parse_distribution_constraints(self, query: str) -> list[str]:
        q = query.lower()
        found: list[str] = []
        for tag in ["CI-H", "CI-M", "HASS-A", "HASS-H", "HASS-S", "REST", "LAB", "PLAB"]:
            if tag.lower() in q:
                found.append(tag)
        for alias, canonical in _DIST_ALIASES.items():
            if alias in q and canonical not in found:
                found.append(canonical)
        return list(dict.fromkeys(found))

    def _matches_distribution(self, course: dict, required_tags: list[str]) -> bool:
        course_dists = [d.upper() for d in course.get("distributions", [])]
        for tag in required_tags:
            if tag == "HASS":
                if not any(d.startswith("HASS") for d in course_dists):
                    return False
            elif tag not in course_dists:
                return False
        return True

    def _bm25_score(self, query_tokens: list[str], doc_idx: int) -> float:
        tf_norm = self._tf_norms[doc_idx]
        return sum(
            self._idf.get(term, 0.0) * tf_norm.get(term, 0.0)
            for term in query_tokens
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        require_distributions: Optional[list[str]] = None,
        level_filter: Optional[str] = None,
        exclude_numbers: Optional[list[str]] = None,
        boost_distributions: Optional[list[str]] = None,
        offered_hint: Optional[str] = None,
    ) -> list[dict]:
        """
        Return the top_k most relevant courses for the query.

        Args:
            query: Free-text query.
            top_k: Max courses to return.
            require_distributions: Hard-filter to courses with ALL listed tags.
                                   Inferred from query text if None.
            level_filter: "undergraduate" | "graduate" | None (both).
            exclude_numbers: Course numbers to skip (already taken/in-progress).
            boost_distributions: Distribution tags to score-boost (remaining reqs).
        """
        exclude_set = {n.upper() for n in (exclude_numbers or [])}
        sem = (offered_hint or "").strip().lower()
        dist_constraints = (
            require_distributions
            if require_distributions is not None
            else self._parse_distribution_constraints(query)
        )

        def _matches_semester(course: dict) -> bool:
            if not sem:
                return True
            offered = (course.get("offered") or "").lower()
            if not offered:
                return True
            if sem == "fall":
                return ("fall" in offered) or re.search(r"\bF\b", offered) is not None
            if sem == "spring":
                return ("spring" in offered) or re.search(r"\bS\b", offered) is not None
            return True

        candidates = [
            (idx, course) for idx, course in enumerate(self.courses)
            if course.get("number", "").upper() not in exclude_set
            and (not dist_constraints or self._matches_distribution(course, dist_constraints))
            and (not level_filter or course.get("level", "").lower() == level_filter.lower())
            and _matches_semester(course)
        ]

        if not candidates:
            # Relax distribution constraint but keep exclude/level
            candidates = [
                (idx, course) for idx, course in enumerate(self.courses)
                if course.get("number", "").upper() not in exclude_set
                and (not level_filter or course.get("level", "").lower() == level_filter.lower())
                and _matches_semester(course)
            ]

        query_tokens = _tokenize(query)

        def score(idx: int, course: dict) -> float:
            s = self._bm25_score(query_tokens, idx)
            # Boost courses satisfying outstanding requirements
            if boost_distributions:
                course_dists = [d.upper() for d in course.get("distributions", [])]
                for tag in boost_distributions:
                    if tag in course_dists:
                        s *= 1.3
            # Slight boost for courses with strong eval ratings
            eval_data = course.get("eval", {})
            if eval_data:
                overall = eval_data.get("overall_rating")
                if overall and overall >= 6.0:
                    s *= 1.1
            return s

        scored = sorted(
            ((score(idx, course), course) for idx, course in candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        return [course for _, course in scored[:top_k]]

    def get_by_number(self, number: str) -> Optional[dict]:
        """Exact lookup by course number (case-insensitive)."""
        number = number.strip().upper()
        for course in self.courses:
            if course.get("number", "").upper() == number:
                return course
        return None

    def get_distributions(self, tag: str) -> list[dict]:
        """All courses satisfying a specific distribution tag."""
        return [
            c for c in self.courses
            if tag in [d.upper() for d in c.get("distributions", [])]
        ]

    @property
    def total_courses(self) -> int:
        return len(self.courses)