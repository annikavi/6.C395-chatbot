"""
BM25-based retrieval over the MIT course catalog (data/courses.json).

BM25 normalises for document length and saturates term frequency, giving
better precision than raw TF-IDF for short free-form student queries.
"""

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path


_DATA_PATH = Path(__file__).parent.parent / "data" / "courses.json"

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


class CourseRetriever:
    """
    Usage:
        retriever = CourseRetriever()
        results = retriever.retrieve("CI-H afternoon AI ethics junior 6-3", top_k=8)
    """

    def __init__(self) -> None:
        self.courses: list[dict] = []
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
        self._build_bm25_index()

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
        require_distributions: list[str] | None = None,
        level_filter: str | None = None,
    ) -> list[dict]:
        """
        Return the top_k most relevant courses for the query.

        Args:
            query: Free-text query.
            top_k: Max courses to return.
            require_distributions: Hard-filter to courses with ALL listed tags.
                                   Inferred from query text if None.
            level_filter: "undergraduate" | "graduate" | None (both).
        """
        dist_constraints = (
            require_distributions
            if require_distributions is not None
            else self._parse_distribution_constraints(query)
        )

        candidates = [
            (idx, course) for idx, course in enumerate(self.courses)
            if (not dist_constraints or self._matches_distribution(course, dist_constraints))
            and (not level_filter or course.get("level", "").lower() == level_filter.lower())
        ]

        if not candidates:
            # Relax constraints so the bot can still return something useful
            candidates = list(enumerate(self.courses))

        query_tokens = _tokenize(query)
        scored = sorted(
            ((self._bm25_score(query_tokens, idx), course) for idx, course in candidates),
            key=lambda x: x[0],
            reverse=True,
        )
        return [course for _, course in scored[:top_k]]

    def get_by_number(self, number: str) -> dict | None:
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
