"""
Infer evaluation context from the user prompt so we run only relevant checks per (prompt, response).
"""

import re
from dataclasses import dataclass, field


@dataclass
class PromptContext:
    """Parsed context from a single prompt for conditional checks."""
    semester: str | None = None       # "Fall" | "Spring"
    distributions: list[str] = field(default_factory=list)  # ["CI-H"], ["REST"], etc.
    department: str | None = None     # "8", "6", "17" (number prefix)
    afternoon: bool = False
    small_class: bool = False
    excluded_prereqs: list[str] = field(default_factory=list)  # ["6.031"]
    graduation: bool = False
    advisor: bool = False
    six_uat: bool = False
    course_6: bool = False           # 6-3, Course 6, EECS
    hass_cih: bool = False           # HASS or CI-H mentioned
    vague: bool = False              # underspecified → clarifying question relevant


def parse_prompt_context(prompt: str) -> PromptContext:
    """Extract context from prompt text to decide which checks to run."""
    p = prompt.strip().lower()
    ctx = PromptContext()

    # Semester
    if re.search(r"\bspring\b", p):
        ctx.semester = "Spring"
    if re.search(r"\bfall\b", p):
        ctx.semester = "Fall"  # overwrite if both, fall wins

    # Distributions (canonical tags)
    for tag in ["CI-H", "CI-M", "HASS-A", "HASS-H", "HASS-S", "REST", "LAB", "PLAB"]:
        if tag.lower() in p:
            ctx.distributions.append(tag)
    if re.search(r"\bhass\b", p) and not any(d.startswith("HASS") for d in ctx.distributions):
        ctx.distributions.append("HASS-H")  # generic HASS

    # Department
    for prefix, patterns in [
        ("8", [r"course\s+8", r"i'?m\s+course\s+8", r"\b8\.\d"]),
        ("6", [r"6-3", r"6-1", r"6-2", r"course\s+6", r"eecs", r"6\.\d"]),
        ("17", [r"course\s+17", r"political"]),
        ("18", [r"course\s+18", r"math"]),
        ("24", [r"course\s+24", r"philosophy", r"linguistics"]),
    ]:
        if any(re.search(pat, p) for pat in patterns):
            ctx.department = prefix
            break
    if ctx.department == "6":
        ctx.course_6 = True

    # Afternoon
    ctx.afternoon = bool(re.search(r"afternoon|afternoons|pm\s+only|only\s+afternoon|afternoon\s+only", p))

    # Small class / seminar
    ctx.small_class = bool(re.search(r"seminar|small\s+class|small\s+classes", p))

    # Excluded prereqs: "don't have X", "without X", "haven't taken X", "no X"
    # Match course numbers like 6.031, 4.031, 18.06
    course_num = r"\b(\d+\.\d+\w*)\b"
    if re.search(r"don'?t\s+have|without|haven'?t\s+taken|don'?t\s+have\s+taken|missing|excluded", p):
        for m in re.finditer(course_num, p):
            ctx.excluded_prereqs.append(m.group(1))
    if re.search(r"not\s+6\.\d|no\s+6\.\d", p):
        for m in re.finditer(r"6\.\d+\w*", p):
            ctx.excluded_prereqs.append(m.group(0))

    # Graduation / requirements
    ctx.graduation = bool(re.search(r"graduat(?:e|ion)|6-3\s+(?:grad|requirement)|degree\s+requirement", p))

    # Advisor / counseling
    ctx.advisor = bool(re.search(r"advisor|who\s+(?:can|do)\s+I\s+talk|who\s+to\s+talk|counseling|struggling|help\s+with", p))

    # 6.UAT
    ctx.six_uat = bool(re.search(r"6\.uat|6uat|under(?:grad)?\s+thesis", p))

    # HASS / CI-H (for mentions_hass_or_cih and have_ci_h_or_hass)
    ctx.hass_cih = bool(re.search(r"ci-h|cih|hass", p)) or bool(ctx.distributions)

    # Vague / underspecified
    ctx.vague = len(p.split()) <= 5 or bool(re.search(r"something|recommend\s+(?:a|some)?\s*$|what'?s\s+good|what\s+about", p))

    return ctx
