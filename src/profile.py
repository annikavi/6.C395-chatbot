"""
Student profile — tracks what the student has told us about themselves.

The profile is extracted from conversation history by a lightweight LLM pass
and persisted in-memory for the session. It is injected into every system
prompt so the advisor always has full context without the student repeating
themselves.

Profile fields:
    year            : "freshman" | "sophomore" | "junior" | "senior" | "graduate" | None
    major           : e.g. "6-3", "Course 18", "TPP" | None
    completed       : list of course numbers already taken, e.g. ["18.01", "6.1010"]
    in_progress     : list of course numbers currently enrolled in
    interests       : free-text keywords the student has expressed interest in
    requirements_remaining : list of distribution tags still needed, e.g. ["CI-H", "HASS-A"]
    constraints     : free-text schedule/workload preferences
    program_context : any special program info (TPP, LGO, etc.)
    gpa_context     : optional difficulty preference
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class StudentProfile:
    year: Optional[str] = None
    major: Optional[str] = None
    semester: Optional[str] = None  # "fall" | "spring" | None
    completed: list[str] = field(default_factory=list)
    in_progress: list[str] = field(default_factory=list)
    interests: list[str] = field(default_factory=list)
    requirements_remaining: list[str] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    program_context: Optional[str] = None
    gpa_context: Optional[str] = None

    def is_empty(self) -> bool:
        return all([
            not self.year, not self.major, not self.semester,
            not self.completed, not self.in_progress,
            not self.interests, not self.requirements_remaining,
            not self.constraints, not self.program_context,
        ])

    def to_prompt_block(self) -> str:
        """Format the profile as a concise block for injection into prompts."""
        if self.is_empty():
            return "No profile information collected yet."

        lines = []
        if self.year:
            lines.append(f"Year: {self.year}")
        if self.major:
            lines.append(f"Major/Program: {self.major}")
        if self.semester:
            lines.append(f"Target semester: {self.semester}")
        if self.completed:
            lines.append(f"Completed courses: {', '.join(self.completed)}")
        if self.in_progress:
            lines.append(f"Currently taking: {', '.join(self.in_progress)}")
        if self.requirements_remaining:
            lines.append(f"Requirements still needed: {', '.join(self.requirements_remaining)}")
        if self.interests:
            lines.append(f"Expressed interests: {', '.join(self.interests)}")
        if self.constraints:
            lines.append(f"Schedule/workload preferences: {', '.join(self.constraints)}")
        if self.program_context:
            lines.append(f"Special program context: {self.program_context}")
        if self.gpa_context:
            lines.append(f"Difficulty preference: {self.gpa_context}")

        return "\n".join(lines)

    def merge(self, updates: dict) -> None:
        """
        Merge extracted profile updates in-place.
        Lists are unioned (no duplicates), scalars overwrite only if not None.
        """
        def _norm_course(c: str) -> str:
            return c.strip().upper()

        if updates.get("year"):
            self.year = updates["year"]
        if updates.get("major"):
            self.major = updates["major"]
        if updates.get("semester"):
            self.semester = updates["semester"]
        if updates.get("program_context"):
            self.program_context = updates["program_context"]
        if updates.get("gpa_context"):
            self.gpa_context = updates["gpa_context"]

        for course in (updates.get("completed") or []):
            n = _norm_course(course)
            if n and n not in self.completed:
                self.completed.append(n)
                # Remove from in_progress if student says they finished it
                if n in self.in_progress:
                    self.in_progress.remove(n)

        for course in (updates.get("in_progress") or []):
            n = _norm_course(course)
            if n and n not in self.in_progress and n not in self.completed:
                self.in_progress.append(n)

        for req in (updates.get("requirements_remaining") or []):
            r = req.strip().upper()
            if r and r not in self.requirements_remaining:
                self.requirements_remaining.append(r)

        # Remove requirements that are now satisfied
        for req in (updates.get("requirements_satisfied") or []):
            r = req.strip().upper()
            if r in self.requirements_remaining:
                self.requirements_remaining.remove(r)

        for interest in (updates.get("interests") or []):
            i = interest.strip().lower()
            if i and i not in self.interests:
                self.interests.append(i)

        for constraint in (updates.get("constraints") or []):
            c = constraint.strip()
            if c and c not in self.constraints:
                self.constraints.append(c)

    def to_dict(self) -> dict:
        return asdict(self)


# ── Profile extraction prompt ─────────────────────────────────────────────────

PROFILE_EXTRACTION_PROMPT = """\
You are extracting structured student information from a single chat message.
Output ONLY valid JSON — no markdown, no explanation.

Extract ONLY what is explicitly stated or strongly implied in the message.
Do NOT infer or assume anything not mentioned.

Output schema (all fields optional, omit if not mentioned):
{
  "year": "freshman|sophomore|junior|senior|graduate" or null,
  "major": "e.g. 6-3, Course 18, TPP, MEng" or null,
  "completed": ["list of course numbers already taken, e.g. 18.01, 6.1010"],
  "in_progress": ["course numbers currently enrolled in this semester"],
  "requirements_remaining": ["distribution tags still needed: CI-H, CI-M, HASS-A, HASS-H, HASS-S, REST, LAB, HASS"],
  "requirements_satisfied": ["distribution tags the student says they have already completed"],
  "interests": ["subject area keywords, e.g. machine learning, ethics, design"],
  "constraints": ["schedule/workload preferences, e.g. no 9am, light workload, afternoon only"],
  "program_context": "any special program info (TPP, LGO, SDM, MEng, etc.)" or null,
  "gpa_context": "difficulty preference if stated, e.g. manageable workload, challenging" or null
}

Examples:
  "I'm a junior in 6-3, I've taken 6.1010 and 18.06, still need my CI-H"
  → {"year":"junior","major":"6-3","completed":["6.1010","18.06"],"requirements_remaining":["CI-H"]}

  "looking for afternoon classes, not too heavy"
  → {"constraints":["afternoon only","light workload"]}

  "I already did my HASS-H last semester"
  → {"requirements_satisfied":["HASS-H"]}

Message to parse:
"""


def extract_profile_updates(message: str, client, model_id: str) -> dict:
    """
    Run a lightweight LLM call to extract profile updates from a message.
    Returns a dict suitable for StudentProfile.merge().
    Falls back to empty dict on any error.
    """
    try:
        resp = client.chat_completion(
            messages=[
                {"role": "user", "content": PROFILE_EXTRACTION_PROMPT + message}
            ],
            max_tokens=300,
            temperature=0.0,
            model=model_id,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```\w*\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
        updates = json.loads(raw)
        print(f"[Profile] Extracted: {json.dumps(updates)}")
        return updates
    except Exception as e:
        print(f"[Profile] Extraction failed: {e}")
        return {}