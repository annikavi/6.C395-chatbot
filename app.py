"""
MIT Course Advisor — Gradio UI
Run with: python app.py
"""

from __future__ import annotations

import re
from datetime import datetime
import traceback
import uuid
import gradio as gr
from typing import Optional

# ── Program dropdown options ───────────────────────────────────────────────────

UNDERGRAD_PROGRAMS = [
    # School of Architecture and Planning
    "Architecture (Course 4)",
    "Art and Design (Course 4-B)",
    "Planning (Course 11)",
    # School of Engineering
    "Aerospace Engineering (Course 16)",
    "Archaeology and Materials (Course 3-C)",
    "Artificial Intelligence and Decision Making (6-4)",
    "Biological Engineering (Course 20)",
    "Chemical-Biological Engineering (Course 10-B)",
    "Chemical Engineering (Course 10)",
    "Chemical Engineering (Course 10-C)",
    "Computer Science and Engineering (Course 6-3)",
    "Electrical Engineering with Computing (Course 6-5)",
    "Engineering (Course 1-ENG)",
    "Engineering (Course 2-A)",
    "Engineering (Course 10-ENG)",
    "Engineering (Course 16-ENG)",
    "Engineering (Course 22-ENG)",
    "Materials Science and Engineering (Course 3)",
    "Materials Science and Engineering (Course 3-A)",
    "Mechanical and Ocean Engineering (Course 2-OE)",
    "Mechanical Engineering (Course 2)",
    "Nuclear Science and Engineering (Course 22)",
    # School of Humanities, Arts, and Social Sciences
    "Anthropology (Course 21A)",
    "Comparative Media Studies (CMS)",
    "Economics (Course 14-1)",
    "Global Studies and Languages (Course 21G)",
    "History (Course 21H)",
    "Humanities (Course 21)",
    "Humanities and Engineering (Course 21E)",
    "Humanities and Science (Course 21S)",
    "Linguistics and Philosophy (Course 24-2)",
    "Literature (Course 21L)",
    "Mathematical Economics (Course 14-2)",
    "Music (Course 21M)",
    "Philosophy (Course 24-1)",
    "Political Science (Course 17)",
    "Science, Technology, and Society/Second Major (STS)",
    "Theater Arts (Course 21T)",
    "Writing (Course 21W)",
    # Sloan School of Management
    "Business Analytics (Course 15-2)",
    "Finance (Course 15-3)",
    "Management (Course 15-1)",
    # School of Science
    "Biology (Course 7)",
    "Brain and Cognitive Sciences (Course 9)",
    "Chemistry (Course 5)",
    "Earth, Atmospheric, and Planetary Sciences (Course 12)",
    "Mathematics (Course 18)",
    "Mathematics with Computer Science (Course 18-C)",
    "Physics (Course 8)",
    # MIT Schwarzman College of Computing
    "Computer Science and Engineering (Course 6-3)",
    "Artificial Intelligence and Decision Making (Course 6-4)",
    "Electrical Engineering with Computing (Course 6-5)",
    # Interdisciplinary Programs
    "Chemistry and Biology (Course 5-7)",
    "Climate System Science and Engineering (Course 1-12)",
    "Computation and Cognition (Course 6-9)",
    "Computer Science and Molecular Biology (Course 6-7)",
    "Computer Science, Economics, and Data Science (Course 6-14)",
    "Urban Science and Planning with Computer Science (Course 11-6)",
]

GRAD_PROGRAMS = [
    # School of Architecture and Planning
    "Architecture (MArch)",
    "Architecture Studies (SMArchS)",
    "Art, Culture, and Technology (SM)",
    "City Planning (SM)",
    # School of Engineering
    "Aeronautics and Astronautics Fields (PhD/ScD)",
    "Biological Engineering (PhD/ScD)",
    "Civil and Environmental Engineering (SM)",
    "Civil and Environmental Engineering (PhD/ScD)",
    "Electrical Engineering and Computer Science (MEng, Course 6-P)",
    "Materials Science and Engineering (PhD/ScD)",
    "Nuclear Science and Engineering (PhD/ScD)",
    "Social and Engineering Systems (PhD/ScD)",
    # School of Humanities, Arts, and Social Sciences
    "Data, Economics, and Design of Policy (MASc)",
    "Economics (PhD)",
    "Linguistics (SM)",
    "Science Writing (SM)",
    # School of Science
    "Brain and Cognitive Sciences Fields (PhD)",
    "Chemistry (PhD)",
    "Earth, Atmospheric, and Planetary Sciences Fields (PhD/ScD)",
    "Mathematics (PhD/ScD)",
    # College of Computing
    "Electrical Engineering and Computer Science (MEng, Course 6-P)",
    "Social and Engineering Systems (PhD/ScD)",
    # Interdisciplinary Programs
    "Biological Oceanography (PhD)",
    "Computation and Cognition (MEng, Course 6-9P)",
    "Computational and Systems Biology (PhD)",
    "Computational Science and Engineering (SM)",
    "Computational Science and Engineering (PhD)",
    "Computer Science and Molecular Biology (MEng, Course 6-7P)",
    "Computer Science, Economics, and Data Science (MEng, Course 6-14P)",
    "Engineering and Management (System Design and Management, SM)",
    "Microbiology (PhD)",
    "Music Technology and Computation (MASc & SM)",
    "Physical Oceanography (PhD)",
    "Real Estate Development (SM)",
    "Statistics (PhD)",
    "Supply Chain Management (MASc & MEng)",
    "Technology and Policy (SM)",
    "Transportation (SM)",
    "Transportation (PhD)",
]


def program_choices_for_year(year: Optional[str]) -> list[str]:
    if (year or "").strip().lower() == "graduate":
        return GRAD_PROGRAMS + ["Other"]
    return UNDERGRAD_PROGRAMS + ["Other"]

# ── Lazy-load the Chatbot so startup errors show clearly ─────────────────────
_init_error: str = ""
_BOTS: dict[str, object] = {}

try:
    from src.chat import Chatbot
    _chatbot_available = True
except Exception as e:
    _chatbot_available = False
    _init_error = traceback.format_exc()
    print(f"[app] ERROR importing Chatbot:\n{_init_error}")


def _get_bot(session_id: str):
    if not _chatbot_available:
        return None
    if session_id not in _BOTS:
        try:
            _BOTS[session_id] = Chatbot()
        except Exception as e:
            print(f"[app] ERROR creating Chatbot: {e}\n{traceback.format_exc()}")
            return None
    return _BOTS[session_id]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _profile_html(profile_text: str) -> str:
    if not profile_text or profile_text.strip() == "No profile information collected yet.":
        return (
            "<div style='color:#888;font-style:italic;padding:8px;font-size:0.85em;'>"
            "No profile yet.<br>Tell me your year, major, and courses taken!</div>"
        )
    rows = []
    for line in profile_text.strip().splitlines():
        if ":" in line:
            label, _, value = line.partition(":")
            rows.append(
                "<div style='margin-bottom:8px;'>"
                "<span style='color:#888;font-size:0.75em;text-transform:uppercase;"
                "letter-spacing:0.05em;'>" + label.strip() + "</span><br>"
                "<span style='font-size:0.9em;color:#e0e0d8;'>" + value.strip() + "</span>"
                "</div>"
            )
    return "<div style='padding:4px;'>" + "\n".join(rows) + "</div>"


# ── Event handlers ────────────────────────────────────────────────────────────

def new_session() -> str:
    return str(uuid.uuid4())


def _default_semester_label() -> str:
    m = datetime.now().month
    if 2 <= m <= 7:
        return "Spring"
    return "Fall"


def respond(
    message: str,
    history: list,
    session_id: str,
    year: Optional[str],
    major: Optional[str],
    completed_csv: Optional[str],
    reqs_needed: Optional[list[str]],
    time_pref: Optional[str],
    workload_pref: Optional[str],
    max_hours: Optional[float],
    prefs_text: Optional[str],
    semester: Optional[str] = None,
):
    if not message.strip():
        return history, _profile_html(""), "", gr.update(visible=False)

    if not _chatbot_available:
        err = (
            "⚠️ **Chatbot failed to start.** Check your terminal for the error.\n\n"
            f"```\n{_init_error[:600]}\n```"
        )
        return history + [[message, err]], _profile_html(""), "", gr.update(visible=False)

    bot = _get_bot(session_id)
    if bot is None:
        err = (
            "⚠️ **Could not initialise the chatbot.** "
            "Make sure `data/courses.json` exists (run `python -m src.scraper` first) "
            "and your HuggingFace token is set in `config.py`."
        )
        return history + [[message, err]], _profile_html(""), "", gr.update(visible=False)

    try:
        prefs_parts: list[str] = []
        if time_pref and time_pref != "No preference":
            prefs_parts.append(time_pref)
        if workload_pref and workload_pref != "No preference":
            prefs_parts.append(workload_pref)
        if max_hours is not None:
            prefs_parts.append(f"max {int(max_hours)} hours/week outside class")
        if prefs_text and prefs_text.strip():
            prefs_parts.append(prefs_text.strip())

        bot.set_structured_profile(
            year=year,
            major=major,
            semester=semester,
            completed_csv=completed_csv,
            requirements_remaining=reqs_needed or [],
            constraints_text="; ".join(prefs_parts),
        )
    except Exception:
        print("[app] apply_structured_profile failed:\n" + traceback.format_exc())

    try:
        answer, profile_summary = bot.get_response(message, history)
    except Exception as e:
        answer = (
            "⚠️ I hit an internal error while answering. "
            "Please try again in a moment or rephrase your question."
        )
        try:
            profile_summary = bot.profile.to_prompt_block()
        except Exception:
            profile_summary = ""
        print(traceback.format_exc())

    new_history = history + [[message, answer]]
    has_courses = bool(re.findall(r"\b\d+\.\w+\b", answer))
    return new_history, _profile_html(profile_summary), "", gr.update(visible=has_courses)


def reset_conversation(session_id: str):
    bot = _get_bot(session_id)
    if bot is not None:
        try:
            bot.reset_profile()
        except Exception:
            pass
    return [], _profile_html(""), ""


def preview_profile(
    session_id: str,
    year: Optional[str],
    major: Optional[str],
    completed_csv: Optional[str],
    reqs_needed: Optional[list[str]],
    time_pref: Optional[str],
    workload_pref: Optional[str],
    max_hours: Optional[float],
    prefs_text: Optional[str],
    semester: Optional[str] = None,
):
    bot = _get_bot(session_id)
    if bot is None:
        return _profile_html("")
    try:
        prefs_parts: list[str] = []
        if time_pref and time_pref != "No preference":
            prefs_parts.append(time_pref)
        if workload_pref and workload_pref != "No preference":
            prefs_parts.append(workload_pref)
        if max_hours is not None:
            prefs_parts.append(f"max {int(max_hours)} hours/week outside class")
        if prefs_text and prefs_text.strip():
            prefs_parts.append(prefs_text.strip())

        bot.set_structured_profile(
            year=year,
            major=major,
            semester=semester,
            completed_csv=completed_csv or "",
            requirements_remaining=reqs_needed or [],
            constraints_text="; ".join(prefs_parts),
        )
        return _profile_html(bot.profile.to_prompt_block())
    except Exception:
        print("[app] preview_profile failed:\n" + traceback.format_exc())
        return _profile_html("")


def apply_preferences(
    session_id: str,
    year: Optional[str],
    major: Optional[str],
    completed_csv: Optional[str],
    reqs_needed: Optional[list[str]],
    time_pref: Optional[str],
    workload_pref: Optional[str],
    max_hours: Optional[float],
    prefs_text: Optional[str],
    semester: Optional[str],
):
    html = preview_profile(
        session_id, year, major, completed_csv, reqs_needed,
        time_pref, workload_pref, max_hours, prefs_text, semester,
    )
    status = "<div style='color:#9f9;font-size:0.8em;margin-top:4px;'>✓ Preferences applied.</div>"
    return html, status


def remove_course_from_schedule(
    session_id: str,
    course_number: str,
    current_schedule: list[dict],
):
    bot = _get_bot(session_id)
    number = (course_number or "").strip().upper()
    if not number:
        return current_schedule, "<div style='color:#f88;'>Enter a course number to remove.</div>"
    new_schedule = [c for c in (current_schedule or []) if c.get("number", "").upper() != number]
    if bot is not None:
        try:
            if number in bot.profile.in_progress:
                bot.profile.in_progress.remove(number)
        except Exception:
            pass
    return (new_schedule, update_schedule_display(bot, new_schedule)) + sync_schedule_rows(new_schedule)


def update_program_dropdown(year: Optional[str]):
    return gr.update(choices=program_choices_for_year(year), value=None)


_MAX_SCHEDULE_ROWS = 8


def _render_schedule_row(course: Optional[dict]) -> tuple:
    if not course:
        return "", "", False
    num = (course.get("number") or "").strip()
    title = (course.get("title") or "").strip()
    sched = (course.get("schedule") or "See catalog").strip()
    html = (
        "<div style='flex:1;border:1px solid #e5e7eb;border-radius:8px;padding:8px 10px;"
        "background:#ffffff;'>"
        f"<div style='font-weight:600;color:#111827;'>{num} "
        f"<span style='font-weight:400;color:#4b5563;'>— {title}</span></div>"
        f"<div style='font-size:0.8em;color:#6b7280;margin-top:3px;'>{sched}</div>"
        "</div>"
    )
    return html, num, True


def sync_schedule_rows(schedule: list) -> tuple:
    """Returns interleaved: html0, num0, btn0, html1, num1, btn1, ..."""
    schedule = schedule or []
    days = ["M", "T", "W", "R", "F"]

    # Detect conflicts
    conflict_nums: set[str] = set()
    try:
        def _parse_t(piece: str) -> float:
            piece = piece.strip()
            if ":" in piece:
                hh, mm = piece.split(":", 1)
                return int(hh) + int(mm) / 60
            if "." in piece:
                hh, frac = piece.split(".", 1)
                if len(frac) == 2:
                    return int(hh) + int(frac) / 60
            return float(int(piece))

        def _parse_range(text):
            m = re.search(r"(\d{1,2}(?::\d{2}|\.\d{2})?)\s*-\s*(\d{1,2}(?::\d{2}|\.\d{2})?)", text)
            if not m: return None
            return _parse_t(m.group(1)), _parse_t(m.group(2))

        events = {d: [] for d in days}
        for c in schedule:
            num = (c.get("number") or "").strip().upper()
            for sec in (c.get("schedule") or "").split("|"):
                md = re.search(r"\b([MTWRF]+)\b", sec)
                rng = _parse_range(sec)
                if md and rng:
                    for d in md.group(1):
                        if d in days:
                            events[d].append((rng[0], rng[1], num))
        for evs in events.values():
            evs.sort()
            for i in range(len(evs)):
                for j in range(i + 1, len(evs)):
                    if evs[j][0] >= evs[i][1]: break
                    conflict_nums.add(evs[i][2]); conflict_nums.add(evs[j][2])
    except Exception:
        pass

    outs = []
    for i in range(_MAX_SCHEDULE_ROWS):
        c = schedule[i] if i < len(schedule) else None
        html, num, visible = _render_schedule_row(c)
        if num and num.upper() in conflict_nums:
            html = (
                "<div style='display:flex;gap:8px;align-items:center;'>"
                + html
                + "<div style='color:#dc2626;border:1px solid #fca5a5;border-radius:999px;"
                "padding:2px 8px;font-size:0.75em;white-space:nowrap;'>CONFLICT</div></div>"
            )
        outs.extend([html, num, gr.update(visible=visible)])
    return tuple(outs)


# ── FIX 1: Calendar with proper rowspan so courses span their full duration ───

def update_schedule_display(bot, schedule: list) -> str:
    if not schedule:
        return (
            "<div style='color:#aaa;font-style:italic;padding:10px 8px;font-size:0.9em;'>"
            "No courses in your schedule yet. Use the box below to add a course by "
            "number (e.g. 6.3900), or ask in chat and then add it.</div>"
        )

    days = ["M", "T", "W", "R", "F"]
    day_labels = {"M": "Mon", "T": "Tue", "W": "Wed", "R": "Thu", "F": "Fri"}
    # 30-minute slots from 8:00 to 18:00
    SLOT_SIZE = 0.5
    slots = [8.0 + SLOT_SIZE * i for i in range(20)]  # 8:00–17:30

    def slot_label(h: float) -> str:
        hh = int(h)
        mm = int(round((h - hh) * 60))
        suffix = "am" if hh < 12 else "pm"
        disp = hh if hh <= 12 else hh - 12
        if disp == 0: disp = 12
        return f"{disp}:{mm:02d} {suffix}"

    def parse_time(piece: str) -> float:
        piece = piece.strip()
        if ":" in piece:
            hh, mm = piece.split(":", 1)
            return int(hh) + int(mm) / 60
        if "." in piece:
            hh, frac = piece.split(".", 1)
            if len(frac) == 2:
                return int(hh) + int(frac) / 60
        return float(int(piece))

    def parse_range(text: str):
        m = re.search(
            r"(\d{1,2}(?::\d{2}|\.\d{2})?)\s*-\s*(\d{1,2}(?::\d{2}|\.\d{2})?)",
            text,
        )
        if not m: return None
        return parse_time(m.group(1)), parse_time(m.group(2))

    # Detect conflicts
    conflict_nums: set[str] = set()
    all_events: list[tuple] = []  # (start, end, day, num, label_html)
    for c in schedule:
        num = (c.get("number") or "").strip().upper()
        title = c.get("title", "")
        sched_str = c.get("schedule", "") or ""
        for sec in sched_str.split("|"):
            md = re.search(r"\b([MTWRF]+)\b", sec)
            rng = parse_range(sec)
            if not md or not rng: continue
            for d in md.group(1):
                if d in days:
                    all_events.append((rng[0], rng[1], d, num, title))

    # collision check
    by_day: dict[str, list] = {d: [] for d in days}
    for ev in all_events:
        by_day[ev[2]].append(ev)
    for evs in by_day.values():
        evs.sort()
        for i in range(len(evs)):
            for j in range(i + 1, len(evs)):
                if evs[j][0] >= evs[i][1]: break
                conflict_nums.add(evs[i][3]); conflict_nums.add(evs[j][3])

    # Build per-cell data: grid[(slot_idx, day)] = (num, title, rowspan, is_first_slot)
    # For each event, find the first slot it covers and compute rowspan.
    slot_index = {s: i for i, s in enumerate(slots)}

    # grid[(slot_idx, day)] -> list of (num, title, rowspan, is_top)
    # We mark only the FIRST slot of each event as is_top=True with the correct rowspan.
    # Subsequent slots of the same event are marked is_top=False (skipped in render).
    grid: dict[tuple, list] = {}
    occupied: set[tuple] = set()  # (slot_idx, day) cells that are already rowspan-covered

    for start, end, day, num, title in all_events:
        # Find first slot >= start
        first_idx = None
        for i, s in enumerate(slots):
            if s >= start - 0.01:
                first_idx = i
                break
        if first_idx is None:
            continue
        # Count how many slots this event spans
        span = 0
        for s in slots[first_idx:]:
            if s + SLOT_SIZE <= end + 0.01:
                span += 1
            else:
                break
        span = max(span, 1)

        is_conflict = num in conflict_nums
        badge = (
            " <span style='font-size:0.7em;color:#dc2626;border:1px solid #fca5a5;"
            "border-radius:999px;padding:1px 5px;'>conflict</span>"
            if is_conflict else ""
        )
        label_html = (
            f"<div style='font-weight:600;font-size:0.85em;'>{num}{badge}</div>"
            f"<div style='font-size:0.75em;color:#4b5563;margin-top:2px;'>{title}</div>"
        )

        key = (first_idx, day)
        grid.setdefault(key, []).append((label_html, span))
        for k in range(first_idx, first_idx + span):
            if k != first_idx:
                occupied.add((k, day))

    # Build HTML with rowspan
    html_parts = [
        "<div style='font-size:0.78em;color:#6b7280;margin-bottom:6px;'>"
        "Each block spans the course's actual meeting time. "
        "Always confirm exact schedules in the official catalog.</div>",
        "<div style='overflow-x:auto;border-radius:8px;border:1px solid #e5e7eb;'>",
        "<table style='width:100%;border-collapse:collapse;font-size:0.84em;background:#fff;table-layout:fixed;'>",
        "<colgroup><col style='width:70px'>",
        *[f"<col style='width:{int(100/5)}%'>" for _ in days],
        "</colgroup>",
        "<thead><tr>",
        "<th style='padding:6px 8px;text-align:left;background:#f9fafb;color:#6b7280;"
        "font-weight:500;border-bottom:2px solid #e5e7eb;font-size:0.82em;'>Time</th>",
    ]
    for d in days:
        html_parts.append(
            f"<th style='padding:6px 4px;text-align:center;background:#f9fafb;"
            f"color:#111827;font-weight:600;border-bottom:2px solid #e5e7eb;"
            f"border-left:1px solid #e5e7eb;'>{day_labels[d]}</th>"
        )
    html_parts.append("</tr></thead><tbody>")

    for si, slot in enumerate(slots):
        html_parts.append("<tr>")
        # Time label only on the hour
        if abs(slot - round(slot)) < 0.01:
            html_parts.append(
                f"<td style='padding:4px 6px;color:#9ca3af;font-size:0.78em;"
                f"border-bottom:1px solid #f3f4f6;white-space:nowrap;"
                f"vertical-align:top;background:#fafafa;'>{slot_label(slot)}</td>"
            )
        else:
            html_parts.append(
                "<td style='padding:0 6px;border-bottom:1px solid #f3f4f6;background:#fafafa;'></td>"
            )

        for d in days:
            if (si, d) in occupied:
                # This cell is covered by a rowspan above — skip it
                continue
            cell_events = grid.get((si, d), [])
            if cell_events:
                # Use the max rowspan among events starting here
                max_span = max(span for _, span in cell_events)
                inner = "".join(
                    f"<div style='margin-bottom:4px;padding:5px 7px;border-radius:6px;"
                    f"background:#eff6ff;border:1px solid #bfdbfe;'>{lbl}</div>"
                    for lbl, _ in cell_events
                )
                html_parts.append(
                    f"<td rowspan='{max_span}' style='padding:4px 5px;vertical-align:top;"
                    f"border-bottom:1px solid #e5e7eb;border-left:1px solid #e5e7eb;"
                    f"background:#fff;'>{inner}</td>"
                )
            else:
                html_parts.append(
                    "<td style='padding:4px 5px;border-bottom:1px solid #f3f4f6;"
                    "border-left:1px solid #e5e7eb;'></td>"
                )
        html_parts.append("</tr>")

    html_parts.append("</tbody></table></div>")
    return "".join(html_parts)


# ── FIX 2: Semester mismatch warning in add_course_to_schedule ────────────────

def _semester_mismatch_warning(course: dict, target_sem: str) -> str:
    """
    Returns a warning HTML string if the course is not offered in target_sem, else "".
    Checks the 'offered' / 'terms' field on the course dict.
    """
    if not target_sem:
        return ""
    offered = (
        course.get("offered") or
        course.get("terms") or
        course.get("term") or
        ""
    ).lower()
    if not offered:
        return ""

    target = target_sem.lower()
    in_fall   = "fall"   in offered or bool(re.search(r"\bfa\b|\bfall\b", offered))
    in_spring = "spring" in offered or bool(re.search(r"\bsp\b|\bspring\b", offered))

    if target == "fall" and in_spring and not in_fall:
        return (
            "<div style='color:#b45309;background:#fffbeb;border:1px solid #fde68a;"
            "border-radius:6px;padding:6px 10px;font-size:0.85em;margin-bottom:6px;'>"
            f"⚠️ <strong>{course.get('number','')}</strong> is typically a "
            "<strong>Spring</strong> subject but your target semester is <strong>Fall</strong>. "
            "It may not be available.</div>"
        )
    if target == "spring" and in_fall and not in_spring:
        return (
            "<div style='color:#b45309;background:#fffbeb;border:1px solid #fde68a;"
            "border-radius:6px;padding:6px 10px;font-size:0.85em;margin-bottom:6px;'>"
            f"⚠️ <strong>{course.get('number','')}</strong> is typically a "
            "<strong>Fall</strong> subject but your target semester is <strong>Spring</strong>. "
            "It may not be available.</div>"
        )
    return ""


def add_course_to_schedule(
    session_id: str,
    course_number: str,
    current_schedule: list[dict],
    semester: str = "",
):
    bot = _get_bot(session_id)
    if bot is None:
        schedule = current_schedule or []
        return (schedule, update_schedule_display(None, schedule)) + sync_schedule_rows(schedule)

    number = (course_number or "").strip().upper()
    if not number:
        schedule = current_schedule or []
        msg = ("<div style='color:#f88;font-size:0.9em;margin:6px 0;'>"
               "Enter a course number first (e.g. 6.3900).</div>")
        return (schedule, msg + update_schedule_display(bot, schedule)) + sync_schedule_rows(schedule)

    try:
        course = bot.retriever.get_by_number(number)
    except Exception:
        print("[app] retriever.get_by_number failed:\n" + traceback.format_exc())
        course = None

    if not course:
        schedule = current_schedule or []
        msg = (
            f"<div style='color:#dc2626;background:#fef2f2;border:1px solid #fca5a5;"
            f"border-radius:6px;padding:6px 10px;font-size:0.85em;margin-bottom:6px;'>"
            f"<strong>{number}</strong> wasn't found in the catalog — it may not be offered "
            f"this semester, or the number may be slightly off. "
            f"<a href='https://student.mit.edu/catalog/' target='_blank' style='color:#2563eb;'>"
            f"Check the MIT catalog.</a></div>"
        )
        return (schedule, msg + update_schedule_display(bot, schedule)) + sync_schedule_rows(schedule)

    # Deduplicate
    if any(c.get("number", "").upper() == number for c in current_schedule):
        schedule = current_schedule
        return (schedule, update_schedule_display(bot, schedule)) + sync_schedule_rows(schedule)

    schedule = current_schedule + [course]

    try:
        if number not in bot.profile.in_progress and number not in bot.profile.completed:
            bot.profile.in_progress.append(number)
    except Exception:
        pass

    # Check semester mismatch
    warning_html = ""
    try:
        warning_html = _semester_mismatch_warning(course, semester or "")
    except Exception:
        pass

    display = warning_html + update_schedule_display(bot, schedule)
    return (schedule, display) + sync_schedule_rows(schedule)


def add_last_suggested_to_schedule(
    session_id: str,
    history: list,
    current_schedule: list[dict],
):
    bot = _get_bot(session_id)
    if bot is None:
        return (current_schedule, "<div style='color:#f88;'>Chatbot not available.</div>") + sync_schedule_rows(current_schedule)

    last_assistant = ""
    if history:
        last_turn = history[-1]
        if isinstance(last_turn, list) and len(last_turn) == 2:
            last_assistant = (last_turn[1] or "").strip()
        elif isinstance(last_turn, dict) and last_turn.get("role") == "assistant":
            last_assistant = (last_turn.get("content") or "").strip()

    if not last_assistant:
        return (current_schedule, (
            "<div style='color:#aaa;font-size:0.9em;'>"
            "No assistant message to sync.</div>"
        )) + sync_schedule_rows(current_schedule)

    numbers = sorted(set(re.findall(r"\b\d+\.\w+\b", last_assistant)))
    if not numbers:
        return (current_schedule, (
            "<div style='color:#aaa;font-size:0.9em;'>"
            "No course numbers found in the last answer.</div>"
        )) + sync_schedule_rows(current_schedule)

    schedule = current_schedule
    for num in numbers:
        result = add_course_to_schedule(session_id, num, schedule)
        schedule = result[0]

    return (schedule, update_schedule_display(bot, schedule)) + sync_schedule_rows(schedule)


def pick_suggested_courses(history: list):
    last_assistant = ""
    if history:
        last_turn = history[-1]
        if isinstance(last_turn, list) and len(last_turn) == 2:
            last_assistant = (last_turn[1] or "").strip()
        elif isinstance(last_turn, dict) and last_turn.get("role") == "assistant":
            last_assistant = (last_turn.get("content") or "").strip()
    nums = sorted(set(re.findall(r"\b\d+\.\w+\b", last_assistant)))
    return gr.update(choices=nums, value=nums), gr.update(visible=bool(nums))


def add_selected_suggested_courses(
    session_id: str,
    selected_numbers: list[str],
    current_schedule: list[dict],
    semester: str = "",
):
    bot = _get_bot(session_id)
    if bot is None:
        schedule = current_schedule or []
        return (schedule, update_schedule_display(None, schedule)) + sync_schedule_rows(schedule)
    schedule = current_schedule or []
    for num in (selected_numbers or []):
        result = add_course_to_schedule(session_id, num, schedule, semester)
        schedule = result[0]
    return (schedule, update_schedule_display(bot, schedule)) + sync_schedule_rows(schedule)


# ── Styles ────────────────────────────────────────────────────────────────────

CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');

body, .gradio-container {
    font-family: 'Inter', system-ui, sans-serif !important;
    background: #f8f9fa !important;
    color: #111827 !important;
}
.gradio-container label, .gradio-container .block label { color: #374151 !important; font-size: 0.82rem !important; font-weight: 500 !important; }
.gradio-container .tabs, .gradio-container .tabitem { background: #f8f9fa !important; }
.gradio-container .tab-nav button { font-size: 0.9rem !important; font-weight: 500 !important; color: #374151 !important; }
.gradio-container .tab-nav button.selected { color: #1d4ed8 !important; border-bottom-color: #1d4ed8 !important; }

/* Header */
.mit-header { background: #ffffff; border-bottom: 1px solid #e5e7eb; padding: 14px 0 12px; margin-bottom: 16px; }
.mit-header h1 { font-size: 1.15rem !important; font-weight: 700 !important; color: #111827 !important; margin: 0 !important; letter-spacing: -0.01em; }
.mit-header p { font-size: 0.78rem; color: #6b7280; margin: 3px 0 0 0; }

/* Chat */
.chatbot-area { border: 1px solid #e5e7eb !important; border-radius: 12px !important; background: #ffffff !important; }
.chatbot-area .message { font-size: 0.91rem !important; line-height: 1.6 !important; color: #111827 !important; }

/* Sidebar */
.profile-title { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: #374151; margin: 0 0 12px 0; }
.tips-box { background: #f0f9ff; border: 1px solid #bae6fd; border-radius: 10px; padding: 10px 13px; margin-top: 12px; font-size: 0.79rem; color: #374151; line-height: 1.65; }
.tips-box strong { color: #0369a1; }

/* Inputs */
textarea {
    background: #ffffff !important; border: 1px solid #d1d5db !important;
    border-radius: 10px !important; color: #111827 !important;
    font-family: 'Inter', system-ui, sans-serif !important;
    font-size: 0.9rem !important;
}
textarea:focus { border-color: #2563eb !important; box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important; outline: none !important; }
textarea::placeholder { color: #9ca3af !important; }
input, select { color: #111827 !important; }

/* Primary send button — full width, prominent */
.send-btn {
    background: #1d4ed8 !important; border-radius: 10px !important; border: none !important;
    color: #ffffff !important; font-weight: 600 !important; font-size: 0.88rem !important;
    min-height: 44px !important; width: 100% !important;
    transition: background 0.15s !important;
}
.send-btn:hover { background: #1e40af !important; }

/* Reset — subtle, same height */
.reset-btn {
    background: #ffffff !important; border-radius: 10px !important;
    border: 1px solid #d1d5db !important; color: #374151 !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    min-height: 44px !important; width: 100% !important;
}
.reset-btn:hover { border-color: #6b7280 !important; color: #111827 !important; }

/* Action buttons — secondary grey pills */
.action-btn {
    background: #ffffff !important; border-radius: 8px !important;
    border: 1px solid #d1d5db !important; color: #374151 !important;
    font-size: 0.82rem !important; font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.15s !important;
}
.action-btn:hover { background: #f3f4f6 !important; border-color: #9ca3af !important; color: #111827 !important; }

/* Remove button — clearly destructive */
.remove-btn {
    background: #fff1f2 !important; border-radius: 8px !important;
    border: 1px solid #fecdd3 !important; color: #be123c !important;
    font-size: 0.8rem !important; font-weight: 500 !important;
    padding: 6px 14px !important;
    transition: all 0.15s !important;
}
.remove-btn:hover { background: #ffe4e6 !important; border-color: #fb7185 !important; }

/* Add-to-schedule button */
.add-btn {
    background: #1d4ed8 !important; border-radius: 8px !important; border: none !important;
    color: #ffffff !important; font-weight: 600 !important; font-size: 0.85rem !important;
    min-height: 42px !important;
}
.add-btn:hover { background: #1e40af !important; }

/* Strip Gradio's default outer borders/shadows/padding */
.gradio-container { border: none !important; box-shadow: none !important; padding: 12px 16px !important; max-width: 100% !important; }
.gr-panel, .gr-box, .gr-form, footer { border: none !important; box-shadow: none !important; background: transparent !important; }
.block { border: none !important; box-shadow: none !important; }
.contain { border: none !important; box-shadow: none !important; }

/* Schedule table */
table { background-color: #ffffff !important; color: #111827 !important; }
th { background: #f9fafb !important; color: #374151 !important; font-weight: 600 !important; }
td { color: #111827 !important; }
th, td { border-color: #e5e7eb !important; }
"""

STARTERS = [
    "I'm a junior in 6-3. I still need my CI-H. What are good options?",
    "Compare 6.3900 and 6.4100 for someone interested in AI",
    "What HASS-S courses are interesting and not too heavy?",
    "I've done 18.01 and 8.01 — what REST subjects should I take next?",
    "I'm an MEng in Course 6, what grad electives are worth it?",
    "Build me a fall schedule: CI-H + one 6-3 core + an elective",
]

CHIPS_HTML = (
    "<div style='margin:8px 0 4px;display:flex;flex-wrap:wrap;gap:6px;'>"
    + "".join(
        "<span style='background:#f3f4f6;border:1px solid #d1d5db;border-radius:999px;"
        "padding:5px 11px;font-size:0.76em;color:#374151;cursor:pointer;transition:all 0.15s;'"
        " onmouseover=\"this.style.background='#e5e7eb';this.style.borderColor='#9ca3af';this.style.color='#111827'\""
        " onmouseout=\"this.style.background='#f3f4f6';this.style.borderColor='#d1d5db';this.style.color='#374151'\""
        " onclick=\"var t=document.querySelector('textarea');"
        "t.value='" + s.replace("'", "\\'") + "';"
        "t.dispatchEvent(new Event('input',{bubbles:true}))\">"
        + (s[:55] + "…" if len(s) > 55 else s) + "</span>"
        for s in STARTERS
    )
    + "</div>"
)

TIPS_HTML = """<div class="tips-box"><strong>💡 Tips</strong><br>
• Tell me your year, major (e.g. "6-3"), and courses already taken<br>
• Mention which requirements you still need (CI-H, REST, HASS-A…)<br>
• Share preferences: "light workload", "afternoons only", "avoid long gaps"<br>
• I remember everything — no need to repeat yourself</div>"""


# ── Build UI ──────────────────────────────────────────────────────────────────

def build_ui():
    with gr.Blocks(css=CSS, title="MIT Course Advisor") as demo:

        session_id = gr.State("")

        status_banner = ""
        if not _chatbot_available:
            status_banner = (
                "<div style='background:#3a1010;border:1px solid #a31f34;border-radius:8px;"
                "padding:10px 14px;margin-bottom:10px;font-size:0.85em;color:#ff9999;'>"
                "⚠️ Chatbot failed to initialise — check terminal for errors.</div>"
            )

        gr.HTML(
            '<div class="mit-header">'
            '<div style="display:flex;align-items:center;gap:14px;">'
            '<img src="https://download.logo.wine/logo/Massachusetts_Institute_of_Technology/Massachusetts_Institute_of_Technology-Logo.wine.png"'
            ' style="height:48px;width:auto;object-fit:contain;" alt="MIT">'
            '<div>'
            '<h1>Course Advisor</h1>'
            '<p>Powered by the MIT course catalog + real student evaluations</p>'
            '</div></div></div>'
            + status_banner
        )

        with gr.Tabs():
            with gr.Tab("Advisor"):
                with gr.Row():
                    with gr.Column(scale=3):
                        chatbox = gr.Chatbot(
                            value=[],
                            height=520,
                            elem_classes="chatbot-area",
                            show_label=False,
                            bubble_full_width=False,
                            render_markdown=True,
                        )
                        gr.HTML(CHIPS_HTML)
                        add_last_btn = gr.Button(
                            "➕ Add last suggested courses to My Schedule",
                            elem_classes="action-btn",
                            visible=False,
                        )
                        with gr.Accordion("Select courses to add", open=True, visible=False) as add_picker:
                            suggested_courses_cb = gr.CheckboxGroup(choices=[], label="Suggested courses")
                            with gr.Row():
                                add_selected_btn = gr.Button("Add selected to My Schedule", elem_classes="action-btn", size="sm")
                                close_picker_btn = gr.Button("Close", elem_classes="reset-btn", size="sm")
                        with gr.Row():
                            msg_input = gr.Textbox(
                                placeholder="Ask about courses, requirements, schedules…",
                                show_label=False,
                                lines=2,
                                max_lines=5,
                                scale=8,
                            )
                            with gr.Column(scale=1, min_width=110):
                                send_btn  = gr.Button("Send",  elem_classes="send-btn")
                                reset_btn = gr.Button("Reset", elem_classes="reset-btn")

                    with gr.Column(scale=1, min_width=260):
                        gr.HTML('<div class="profile-title">📋 Your current profile</div>')
                        profile_display = gr.HTML(value=_profile_html(""))
                        semester_dd = gr.Dropdown(
                            ["Fall", "Spring"],
                            label="Target semester",
                            value=_default_semester_label(),
                        )
                        year_dd = gr.Dropdown(
                            ["First-year", "Sophomore", "Junior", "Senior", "Graduate"],
                            label="Year",
                            value=None,
                        )
                        major_dd = gr.Dropdown(
                            program_choices_for_year(None),
                            label="Major / Program",
                            value=None,
                        )
                        completed_tb = gr.Textbox(
                            label="Key courses taken (comma-separated)",
                            placeholder="e.g. 6.1000, 6.1010, 18.02",
                            lines=2,
                        )
                        reqs_cb = gr.CheckboxGroup(
                            ["CI-H", "CI-M", "HASS-A", "HASS-H", "HASS-S", "REST", "LAB", "HASS"],
                            label="Requirements still needed",
                        )
                        with gr.Accordion("Preferences (optional)", open=True):
                            time_pref = gr.Dropdown(
                                ["No preference", "morning classes", "afternoon classes", "evening classes", "no Friday"],
                                label="Time preference",
                                value="No preference",
                            )
                            workload_pref = gr.Dropdown(
                                ["No preference", "light workload", "moderate workload", "challenging workload"],
                                label="Workload preference",
                                value="No preference",
                            )
                            max_hours = gr.Slider(
                                minimum=0, maximum=20, value=0, step=1,
                                label="Max hours/week outside class (0 = ignore)",
                            )
                            prefs_tb = gr.Textbox(
                                label="Other constraints (free text)",
                                placeholder="e.g. TR only; avoid finals; prefer project-based",
                                lines=2,
                            )
                            prefs_apply = gr.Button("Apply preferences", elem_classes="action-btn", size="sm")
                            prefs_status = gr.HTML(
                                "<div style='color:#6b7280;font-size:0.8em;margin-top:4px;'>Not applied yet.</div>"
                            )
                        gr.HTML(TIPS_HTML)

            with gr.Tab("My Schedule"):
                schedule_state = gr.State([])
                schedule_html = gr.HTML(update_schedule_display(None, []))
                with gr.Row():
                    add_course_box = gr.Textbox(
                        label="Add course to schedule by number",
                        placeholder="e.g. 6.3900",
                        lines=1,
                    )
                    add_course_btn = gr.Button("Add to schedule", elem_classes="add-btn", size="sm")

                gr.Markdown("Planned courses (click **Remove** to delete):")
                sched_row_htmls = []
                sched_row_nums = []
                sched_row_remove_btns = []
                for i in range(_MAX_SCHEDULE_ROWS):
                    with gr.Row(elem_classes="sched-row"):
                        row_html = gr.HTML("")
                        row_num = gr.Textbox(value="", visible=False)
                        rm_btn = gr.Button("Remove", elem_classes="remove-btn", visible=False, size="sm")
                        sched_row_htmls.append(row_html)
                        sched_row_nums.append(row_num)
                        sched_row_remove_btns.append(rm_btn)

        # ── Helper for interleaved outputs ────────────────────────────────────
        def _row_outputs():
            return [x for t in zip(sched_row_htmls, sched_row_nums, sched_row_remove_btns) for x in t]

        # ── Schedule tab wiring ───────────────────────────────────────────────
        add_course_btn.click(
            fn=add_course_to_schedule,
            inputs=[session_id, add_course_box, schedule_state, semester_dd],
            outputs=[schedule_state, schedule_html] + _row_outputs(),
            api_name=False,
        )
        for i in range(_MAX_SCHEDULE_ROWS):
            sched_row_remove_btns[i].click(
                fn=remove_course_from_schedule,
                inputs=[session_id, sched_row_nums[i], schedule_state],
                outputs=[schedule_state, schedule_html] + _row_outputs(),
                api_name=False,
            )
        demo.load(
            fn=lambda: sync_schedule_rows([]),
            inputs=[],
            outputs=_row_outputs(),
            api_name=False,
        )

        # ── Advisor tab wiring ────────────────────────────────────────────────
        add_last_btn.click(fn=pick_suggested_courses, inputs=[chatbox], outputs=[suggested_courses_cb, add_picker], api_name=False)
        close_picker_btn.click(fn=lambda: gr.update(visible=False), inputs=[], outputs=[add_picker], api_name=False)
        add_selected_btn.click(
            fn=add_selected_suggested_courses,
            inputs=[session_id, suggested_courses_cb, schedule_state, semester_dd],
            outputs=[schedule_state, schedule_html] + _row_outputs(),
            api_name=False,
        )
        add_selected_btn.click(fn=lambda: gr.update(visible=False), inputs=[], outputs=[add_picker], api_name=False)

        _preview_inputs = [session_id, year_dd, major_dd, completed_tb, reqs_cb, time_pref, workload_pref, max_hours, prefs_tb, semester_dd]
        _respond_inputs = [msg_input, chatbox, session_id, year_dd, major_dd, completed_tb, reqs_cb, time_pref, workload_pref, max_hours, prefs_tb, semester_dd]

        for comp in [year_dd, major_dd, reqs_cb, time_pref, workload_pref, max_hours, semester_dd]:
            comp.change(fn=preview_profile, inputs=_preview_inputs, outputs=[profile_display], api_name=False)
        year_dd.change(fn=update_program_dropdown, inputs=[year_dd], outputs=[major_dd], api_name=False)
        for comp in [completed_tb, prefs_tb]:
            comp.blur(fn=preview_profile, inputs=_preview_inputs, outputs=[profile_display], api_name=False)
        prefs_apply.click(fn=apply_preferences, inputs=_preview_inputs, outputs=[profile_display, prefs_status], api_name=False)

        send_btn.click(fn=respond, inputs=_respond_inputs, outputs=[chatbox, profile_display, msg_input, add_last_btn], api_name=False)
        msg_input.submit(fn=respond, inputs=_respond_inputs, outputs=[chatbox, profile_display, msg_input, add_last_btn], api_name=False)
        reset_btn.click(fn=reset_conversation, inputs=[session_id], outputs=[chatbox, profile_display, msg_input], api_name=False)
        demo.load(fn=new_session, inputs=[], outputs=[session_id], api_name=False)

    return demo


if __name__ == "__main__":
    app = build_ui()
    app.launch(share=True)