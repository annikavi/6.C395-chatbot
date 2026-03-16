#!/usr/bin/env python3
"""
MIT Subject Evaluation Scraper (Touchstone-protected)
====================================================
Firefox-only scraper for MIT's Subject Evaluation Reports (OSE).

This module is designed to live inside the chatbot repo and output to:
  6.C395-chatbot/data/evaluations.json

USAGE (recommended):
  python -m src.evaluations_scraper --browser

Test a small run:
  python -m src.evaluations_scraper --browser --limit 5

Notes:
- This requires Firefox. On recent Selenium versions, the driver is often
  managed automatically; if Firefox fails to launch, install GeckoDriver
  and ensure it is on your PATH.
- You will complete Touchstone login (and Duo) in the opened browser window.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import parse_qs, urljoin, urlparse

import requests
from bs4 import BeautifulSoup


# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = "https://eduapps.mit.edu/ose-rpt/"
SEARCH_URL = BASE_URL + "subjectEvaluationSearch.htm"

TERMS = {
    "2026FA": "Fall Term 2025-2026",
    "2025SP": "Spring Term 2024-2025",
}

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; MIT-Eval-Scraper/1.0; educational use)",
    "Accept": "text/html,application/xhtml+xml",
    "Referer": SEARCH_URL,
}

REQUEST_DELAY = 0.75  # seconds between requests

DEFAULT_OUTPUT = Path(__file__).parent.parent / "data" / "evaluations.json"


# ── Browser login (Firefox only) ──────────────────────────────────────────────

def browser_login() -> requests.Session:
    """
    Opens a real Firefox window for MIT Touchstone login, then transfers
    authenticated cookies into a requests.Session.
    """
    try:
        from selenium import webdriver
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.common.by import By
    except ImportError:
        raise SystemExit("Selenium not installed. Run:  pip install selenium")

    print("=" * 60)
    print("Opening Firefox for MIT Touchstone login…")
    print("Log in with your MIT credentials in the browser window.")
    print("The scraper resumes automatically once you are logged in.")
    print("=" * 60 + "\n")

    try:
        driver = webdriver.Firefox()
    except Exception as e:
        print(f"\nERROR: Could not launch Firefox: {e}")
        print("Install GeckoDriver: https://github.com/mozilla/geckodriver/releases")
        raise SystemExit(1)

    driver.get(SEARCH_URL)
    print("Waiting for you to complete login (up to 3 minutes)…")

    try:
        WebDriverWait(driver, 180).until(EC.presence_of_element_located((By.ID, "searchForm")))
    except Exception:
        print("\nERROR: Timed out waiting for login. Please try again.")
        driver.quit()
        raise SystemExit(1)

    print("Login successful! Transferring session cookies…\n")

    session = requests.Session()
    session.headers.update(HEADERS)
    for cookie in driver.get_cookies():
        domain = cookie.get("domain", "eduapps.mit.edu").lstrip(".")
        session.cookies.set(cookie["name"], cookie["value"], domain=domain)

    driver.quit()
    return session


# ── Page fetching ─────────────────────────────────────────────────────────────

def get_soup(session: requests.Session, url: str, params: dict | None = None) -> BeautifulSoup:
    resp = session.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "html.parser")


def check_auth(soup: BeautifulSoup) -> bool:
    """Return False if the page looks like a login/redirect page."""
    text = soup.get_text(separator=" ").lower()
    bad = ["touchstone", "log in to", "kerberos", "please authenticate", "sign in with", "stale request"]
    return not any(b in text for b in bad)


# ── Step 1: Get subject list for a term ───────────────────────────────────────

def get_subject_links(session: requests.Session, term_id: str) -> list[dict]:
    params = {
        "termId": term_id,
        "departmentId": "",
        "subjectCode": "",
        "instructorName": "",
        "search": "Search",
    }

    print(f"  Fetching subject list for term {term_id}…")
    soup = get_soup(session, SEARCH_URL, params=params)

    if not check_auth(soup):
        raise PermissionError(
            "Not authenticated! The cookie/session appears invalid or expired.\n"
            "Try running with --browser to log in via Firefox."
        )

    subjects: list[dict] = []
    seen: set[str] = set()

    for link in soup.find_all("a", href=re.compile(r"subjectEvaluationReport\.htm")):
        href = link.get("href", "")
        url = urljoin(BASE_URL, href)
        if url in seen:
            continue
        seen.add(url)

        qs = parse_qs(urlparse(url).query)
        subject_code = (qs.get("subjectId") or qs.get("subjectCode") or [""])[0]
        subjects.append(
            {
                "subject_code": subject_code,
                "subject_title": link.get_text(strip=True),
                "url": url,
            }
        )

    print(f"  Found {len(subjects)} subjects.")
    return subjects


# ── Step 2: Parse one subject evaluation report ───────────────────────────────

def parse_num(text):
    try:
        return float(str(text).strip().replace("%", ""))
    except (ValueError, TypeError):
        return None


def parse_subject_report(session: requests.Session, url: str) -> dict:
    soup = get_soup(session, url)

    result = {
        "url": url,
        "subject_numbers": [],
        "subject_title": None,
        "term": None,
        "eligible_respondents": None,
        "total_respondents": None,
        "response_rate_pct": None,
        "overall_subject_rating": None,
        "instructors": [],
        "subject_ratings": {},
        "pace": None,
        "hours_per_week_in_class": None,
        "hours_per_week_outside_class": None,
    }

    # Header
    header = soup.find("table", {"class": "header"})
    if header:
        h1 = header.find("h1")
        if h1:
            for line in h1.get_text("\n").strip().splitlines():
                line = line.strip()
                m = re.match(r"^(\\S+)\\s+(.+)$", line)
                if m:
                    result["subject_numbers"].append(m.group(1))
                    if not result["subject_title"]:
                        result["subject_title"] = m.group(2).strip()
        h2 = header.find("h2")
        if h2:
            m = re.search(r"Survey Window:\\s*(.+?)(?:\\s*\\||\\s*$)", h2.get_text(" ", strip=True))
            if m:
                result["term"] = m.group(1).strip()

    # Summary stats
    summary = soup.find("table", {"class": "summary"})
    if summary:
        t = summary.get_text(" ", strip=True)
        for pattern, key, cast in [
            (r"Eligible to Respond:\\s*(\\d+)", "eligible_respondents", int),
            (r"Total # of Respondents:\\s*(\\d+)", "total_respondents", int),
            (r"Response rate:\\s*([\\d.]+)%", "response_rate_pct", float),
        ]:
            m = re.search(pattern, t)
            if m:
                result[key] = cast(m.group(1))

        m = re.search(r"Overall rating of subject:\\s*([\\d.]+)\\s*out of\\s*([\\d.]+)", t)
        if m:
            result["overall_subject_rating"] = {"avg": float(m.group(1)), "out_of": float(m.group(2))}

    # Instructor ratings
    instr_anchor = soup.find("a", {"name": "instructors"})
    if instr_anchor:
        instr_table = instr_anchor.find_next("table", {"class": "grid"})
        if instr_table:
            col_headers = []
            for row in instr_table.find_all("tr"):
                ths = row.find_all("th")
                if any(th.get_text(strip=True) == "NAME" for th in ths):
                    col_headers = [th.get_text(strip=True) for th in ths[1:]]
                    break

            for row in instr_table.find_all("tr"):
                cells = row.find_all("td")
                if not cells:
                    continue
                name_link = cells[0].find("a")
                if not name_link:
                    continue

                name_text = cells[0].get_text(" ", strip=True)
                m = re.match(
                    r"^(.+?),?\\s*(Instructor|Teaching Assistant|Lecturer|Professor|Recitation Instructor)\\s*(?:\\((.+?)\\))?",
                    name_text,
                )
                if m:
                    name = m.group(1).strip()
                    role = m.group(2).strip()
                    section = m.group(3) or ""
                else:
                    name = name_link.get_text(strip=True)
                    role, section = "", ""

                ratings = {}
                for i, cell in enumerate(cells[1:]):
                    avg_span = cell.find("span", {"class": "avg"})
                    val = parse_num(avg_span.get_text(strip=True) if avg_span else cell.get_text(strip=True))
                    label = col_headers[i] if i < len(col_headers) else f"rating_{i+1}"
                    ratings[label] = val

                result["instructors"].append(
                    {
                        "name": name,
                        "role": role,
                        "section_type": section,
                        "ratings": ratings,
                        "instructor_report_url": urljoin(BASE_URL, name_link.get("href", "")),
                    }
                )

    # Subject question ratings
    subj_anchor = soup.find("a", {"name": "subjects"})
    if subj_anchor:
        parent_h2 = subj_anchor.find_parent("h2")
        if parent_h2:
            for table in parent_h2.find_next_siblings("table"):
                for row in table.find_all("tr"):
                    cells = row.find_all("td")
                    if len(cells) < 2:
                        continue
                    question = cells[0].get_text(strip=True)
                    if not question or question == "\\xa0":
                        continue

                    avg_span = cells[1].find("span", {"class": "avg"})
                    avg = parse_num(avg_span.get_text(strip=True) if avg_span else cells[1].get_text(strip=True))
                    responses = parse_num(cells[3].get_text(strip=True)) if len(cells) > 3 else None
                    median = parse_num(cells[4].get_text(strip=True)) if len(cells) > 4 else None
                    stdev = parse_num(cells[5].get_text(strip=True)) if len(cells) > 5 else None
                    entry = {"avg": avg, "responses": responses, "median": median, "stdev": stdev}

                    ql = question.lower()
                    if "hours" in ql and "classroom" in ql:
                        result["hours_per_week_in_class"] = entry
                    elif "hours" in ql and "outside" in ql:
                        result["hours_per_week_outside_class"] = entry
                    elif "pace" in ql:
                        entry["_note"] = "Scale: 1=Too Slow, 4=Just Right, 7=Too Fast"
                        result["pace"] = entry
                    elif "overall rating of the subject" in ql:
                        if result["overall_subject_rating"]:
                            result["overall_subject_rating"].update({"median": median, "stdev": stdev})
                        else:
                            result["overall_subject_rating"] = {"avg": avg, "out_of": 7, "median": median, "stdev": stdev}
                    else:
                        result["subject_ratings"][question] = entry

    return result


# ── Step 3: Orchestrate ───────────────────────────────────────────────────────

def scrape_all(session: requests.Session, limit: int | None = None) -> dict:
    output = {
        "metadata": {
            "source": BASE_URL,
            "terms_scraped": list(TERMS.keys()),
            "term_names": TERMS,
            "description": "MIT Subject Evaluation data for course-selection chatbot",
        },
        "terms": {},
    }

    for term_id, term_name in TERMS.items():
        print(f"\n{'='*60}")
        print(f"Term: {term_name} ({term_id})")
        print(f"{'='*60}")

        subjects = get_subject_links(session, term_id)
        if limit:
            subjects = subjects[:limit]
            print(f"  (Limited to {limit} subjects for testing)")

        term_data = {
            "term_id": term_id,
            "term_name": term_name,
            "total_subjects": len(subjects),
            "courses": [],
        }

        for i, subj in enumerate(subjects, 1):
            label = subj.get("subject_code") or subj.get("subject_title", "")
            print(f"  [{i}/{len(subjects)}] {label}")
            try:
                report = parse_subject_report(session, subj["url"])
                term_data["courses"].append(report)
            except Exception as e:
                print(f"    ERROR: {e}")
                term_data["courses"].append(
                    {"url": subj["url"], "subject_code": subj.get("subject_code"), "error": str(e)}
                )
            time.sleep(REQUEST_DELAY)

        output["terms"][term_id] = term_data
        print(f"\\n  Done: {len(term_data['courses'])} courses scraped for {term_name}")

    return output


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    global REQUEST_DELAY

    parser = argparse.ArgumentParser(description="Scrape MIT Subject Evaluation Reports to JSON")
    parser.add_argument("--browser", action="store_true", help="Open Firefox for Touchstone login (required)")
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help=f"Output JSON file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument("--limit", type=int, default=None, metavar="N", help="Limit to N courses per term (testing)")
    parser.add_argument("--delay", type=float, default=REQUEST_DELAY, metavar="SECONDS", help="Delay between requests")
    args = parser.parse_args()
    REQUEST_DELAY = args.delay

    if not args.browser:
        print("This scraper requires an authenticated Touchstone browser session.")
        print("Run with:  python -m src.evaluations_scraper --browser")
        raise SystemExit(1)

    session = browser_login()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Output file: {output_path}")

    data = scrape_all(session, limit=args.limit)
    output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    total = sum(len(t["courses"]) for t in data["terms"].values())
    print(f"\\nDone! {total} course evaluations saved to: {output_path}")


if __name__ == "__main__":
    main()

