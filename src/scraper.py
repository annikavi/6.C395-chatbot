"""
Fetches MIT course data from Hydrant (hydrant.mit.edu) and writes data/courses.json.

Sources:
    Spring 2026: https://hydrant.mit.edu/latest.json
    Fall 2025:   https://hydrant.mit.edu/f25.json

Usage:
    python -m src.scraper
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import requests


OUTPUT_PATH = Path(__file__).parent.parent / "data" / "courses.json"

HYDRANT_URLS = [
    ("https://hydrant.mit.edu/latest.json", "SP"),
    ("https://hydrant.mit.edu/f25.json",    "FA"),
]

DEPARTMENT_MAP: dict[str, str] = {
    "1":   "Civil and Environmental Engineering",
    "2":   "Mechanical Engineering",
    "3":   "Materials Science and Engineering",
    "4":   "Architecture",
    "5":   "Chemistry",
    "6":   "EECS",
    "7":   "Biology",
    "8":   "Physics",
    "9":   "Brain and Cognitive Sciences",
    "10":  "Chemical Engineering",
    "11":  "Urban Studies and Planning",
    "12":  "Earth, Atmospheric and Planetary Sciences",
    "14":  "Economics",
    "15":  "Management",
    "16":  "Aeronautics and Astronautics",
    "17":  "Political Science",
    "18":  "Mathematics",
    "20":  "Biological Engineering",
    "21A": "Anthropology",
    "21G": "Global Languages",
    "21H": "History",
    "21L": "Literature",
    "21M": "Music and Theater Arts",
    "21T": "Music and Theater Arts",
    "21W": "Writing",
    "22":  "Nuclear Science and Engineering",
    "24":  "Linguistics and Philosophy",
    "CC":  "Concourse",
    "CMS": "Comparative Media Studies",
    "CSB": "Computational and Systems Biology",
    "EC":  "Edgerton Center",
    "ES":  "Experimental Study Group",
    "HST": "Health Sciences and Technology",
    "IDS": "Institute for Data, Systems, and Society",
    "MAS": "Media Arts and Sciences",
    "NS":  "Naval Science",
    "PE":  "Physical Education",
    "RES": "Resources",
    "SCM": "Supply Chain Management",
    "SP":  "Special Programs",
    "STS": "Science, Technology, and Society",
    "WGS": "Women and Gender Studies",
}

HASS_MAP = {"A": "HASS-A", "H": "HASS-H", "S": "HASS-S", "E": "HASS"}
TERM_MAP  = {"FA": "Fall", "SP": "Spring", "JA": "IAP", "SU": "Summer"}


def _department(course_prefix: str) -> str:
    return DEPARTMENT_MAP.get(course_prefix, f"Course {course_prefix}")


def _distributions(h: dict) -> list[str]:
    dists: list[str] = []
    for tag in h.get("hass", []):
        d = HASS_MAP.get(tag)
        if d:
            dists.append(d)
    comms = h.get("comms", "")
    if comms in ("CI-H", "CI-HW"):
        dists.append("CI-H")
    elif comms == "CI-M":
        dists.append("CI-M")
    gir = h.get("gir", "")
    if gir == "REST":
        dists.append("REST")
    elif gir == "LAB":
        dists.append("LAB")
    elif gir == "LAB2":
        dists.extend(["LAB", "PLAB"])
    return dists


def _schedule(h: dict) -> str:
    def fmt(raw_list: list[str], label: str) -> str:
        parts = []
        for entry in raw_list[:3]:
            if entry == "TBA":
                parts.append("TBA")
                continue
            pieces = entry.split("/")
            if len(pieces) < 4:
                parts.append(entry)
                continue
            room, days, _, time = pieces[0], pieces[1], pieces[2], pieces[3]
            parts.append(f"{days} {time} ({room})")
        return f"{label}: {'; '.join(parts)}" if parts else ""

    sections = []
    if h.get("lectureRawSections"):
        s = fmt(h["lectureRawSections"], "Lec")
        if s:
            sections.append(s)
    if h.get("recitationRawSections"):
        s = fmt(h["recitationRawSections"][:2], "Rec")
        if s:
            sections.append(s)
    if h.get("labRawSections"):
        s = fmt(h["labRawSections"][:2], "Lab")
        if s:
            sections.append(s)
    return " | ".join(sections)


def _convert(h: dict) -> dict:
    lu  = h.get("lectureUnits", 0)
    lbu = h.get("labUnits", 0)
    pu  = h.get("preparationUnits", 0)
    units = "Variable" if h.get("isVariableUnits") else f"{lu}-{lbu}-{pu}"
    terms = [TERM_MAP.get(t, t) for t in h.get("terms", [])]

    entry: dict = {
        "number":        h.get("number", ""),
        "title":         h.get("name", ""),
        "units":         units,
        "prereqs":       h.get("prereqs") or "None",
        "distributions": _distributions(h),
        "description":   h.get("description", ""),
        "schedule":      _schedule(h),
        "instructors":   h.get("inCharge") or "See catalog",
        "level":         "graduate" if h.get("level") == "G" else "undergraduate",
        "offered":       "/".join(terms) if terms else "See catalog",
        "department":    _department(h.get("course", "")),
    }

    if rating := h.get("rating"):
        entry["rating"] = round(rating, 1)
    if hours := h.get("hours"):
        entry["avg_hours"] = round(hours, 1)
    if size := h.get("size"):
        entry["avg_class_size"] = round(size, 0)
    if same := h.get("same", ""):
        entry["same_as"] = same
    if meets := h.get("meets", ""):
        entry["meets_with"] = meets
    if cim := h.get("cim", []):
        entry["ci_m_for"] = cim
    if h.get("limited"):
        entry["limited_enrollment"] = True
    if h.get("new"):
        entry["new_course"] = True
    if h.get("half"):
        entry["half_semester"] = True
    if h.get("final"):
        entry["has_final"] = True
    if h.get("virtualStatus"):
        entry["virtual"] = True

    return entry


def scrape_catalog() -> None:
    """Fetch Spring + Fall Hydrant data, merge, and write data/courses.json."""
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[str, dict] = {}
    for url, _ in HYDRANT_URLS:
        print(f"Fetching {url} ...")
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        classes: dict = resp.json().get("classes", {})
        print(f"  → {len(classes)} courses")

        for num, h in classes.items():
            if num not in merged:
                merged[num] = h
            else:
                existing = set(merged[num].get("terms", []))
                merged[num]["terms"] = list(existing | set(h.get("terms", [])))

    courses = sorted([_convert(h) for h in merged.values()], key=lambda c: c["number"])

    output = {
        "courses":    courses,
        "scraped_at": date.today().isoformat(),
        "sources":    [url for url, _ in HYDRANT_URLS],
        "total":      len(courses),
    }

    with OUTPUT_PATH.open("w") as f:
        json.dump(output, f, indent=2)

    print(f"\nDone. Saved {len(courses)} courses to {OUTPUT_PATH}")


if __name__ == "__main__":
    scrape_catalog()
