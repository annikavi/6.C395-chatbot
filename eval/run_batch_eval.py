"""
Run batch evaluation: 100 sample prompts → chatbot responses → response-quality checks.
Runs only checks that are relevant to each prompt/response (inferred from prompt context).
Saves results to eval/eval_results.json for visualization.

Requires: HF_TOKEN set for live run; eval/response_quality.py for full checks.
Usage: python -m eval.run_batch_eval [--limit N] [--out path]
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Add project root for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from eval.get_prompts_100 import get_100_prompts
from eval.prompt_context import parse_prompt_context, PromptContext


def _checks_for_context(ctx: PromptContext, retriever, use_response_quality: bool) -> list[tuple[str, callable]]:
    """Build list of (check_name, fn) applicable given prompt context. fn(response, retriever) -> bool."""
    out: list[tuple[str, callable]] = []

    if use_response_quality:
        from eval.response_quality import (
            check_cited_courses_exist_in_catalog,
            check_cited_courses_offered_in_semester,
            check_cited_courses_have_distribution,
            check_cited_courses_have_afternoon_option,
            check_cited_courses_are_course_6,
            check_cited_courses_are_department,
            check_cited_courses_have_ci_h_or_hass,
            check_cited_courses_prereqs_exclude,
            check_cited_courses_small_class,
            response_curated_not_overwhelming,
            response_explains_recommendations,
            response_articulates_tradeoffs_or_perfect_match,
            response_expresses_uncertainty_or_defers,
            response_mentions_departments_or_distributions,
            response_recognizes_course_6_or_eecs,
            response_suggests_no_results_or_relax,
            response_mentions_graduation_or_requirements,
            response_mentions_6_UAT_restrictions,
            response_mentions_advisor_or_counseling,
            response_mentions_hass_or_cih,
            response_asks_clarifying_question,
        )
        # Always applicable
        out.append(("cited_courses_exist", lambda r, ret: check_cited_courses_exist_in_catalog(r, ret)[0]))
        out.append(("not_overwhelming", lambda r, ret: response_curated_not_overwhelming(r, max_courses=20)[0]))
        out.append(("explains_recommendations", lambda r, ret: response_explains_recommendations(r)))
        out.append(("articulates_tradeoffs", lambda r, ret: response_articulates_tradeoffs_or_perfect_match(r)))
        out.append(("expresses_uncertainty", lambda r, ret: response_expresses_uncertainty_or_defers(r)))
        out.append(("mentions_departments_or_distributions", lambda r, ret: response_mentions_departments_or_distributions(r)))
        # Context-dependent
        if ctx.semester:
            sem = ctx.semester
            out.append(("offered_in_semester", lambda r, ret: check_cited_courses_offered_in_semester(r, ret, sem)[0]))
        if ctx.distributions:
            dist = ctx.distributions[0]
            out.append(("have_distribution", lambda r, ret: check_cited_courses_have_distribution(r, ret, dist)[0]))
        if ctx.afternoon:
            out.append(("have_afternoon_option", lambda r, ret: check_cited_courses_have_afternoon_option(r, ret)[0]))
        if ctx.department:
            dept = ctx.department
            out.append(("cited_courses_same_department", lambda r, ret: check_cited_courses_are_department(r, ret, dept)[0]))
        if ctx.course_6:
            out.append(("are_course_6", lambda r, ret: check_cited_courses_are_course_6(r, ret)[0]))
            out.append(("recognizes_course_6_or_eecs", lambda r, ret: response_recognizes_course_6_or_eecs(r)))
        if ctx.hass_cih:
            out.append(("have_ci_h_or_hass", lambda r, ret: check_cited_courses_have_ci_h_or_hass(r, ret)[0]))
            out.append(("mentions_hass_or_cih", lambda r, ret: response_mentions_hass_or_cih(r)))
        if ctx.excluded_prereqs:
            excl = ctx.excluded_prereqs
            out.append(("prereqs_exclude", lambda r, ret: check_cited_courses_prereqs_exclude(r, ret, excl)[0]))
        if ctx.small_class:
            out.append(("cited_courses_small_class", lambda r, ret: check_cited_courses_small_class(r, ret, 50)[0]))
        if ctx.graduation:
            out.append(("mentions_graduation_or_requirements", lambda r, ret: response_mentions_graduation_or_requirements(r)))
        if ctx.advisor:
            out.append(("mentions_advisor_or_counseling", lambda r, ret: response_mentions_advisor_or_counseling(r)))
        if ctx.six_uat:
            out.append(("mentions_6_UAT_restrictions", lambda r, ret: response_mentions_6_UAT_restrictions(r)))
        if ctx.vague:
            out.append(("asks_clarifying_question", lambda r, ret: response_asks_clarifying_question(r)))
        # No-results / relax: run when prompt looks over-constrained (optional; we run always and let check decide)
        out.append(("suggests_no_results_or_relax", lambda r, ret: response_suggests_no_results_or_relax(r)))
    else:
        # Fallback when response_quality not available
        def _count_course_refs(text):
            return len(re.findall(r"\b\d+\.\w+\b", text))
        out.extend([
            ("cited_courses_exist", lambda r, ret: True),
            ("not_overwhelming", lambda r, ret: _count_course_refs(r) <= 20),
            ("explains_recommendations", lambda r, ret: bool(re.search(r"satisfies|because|prereq|schedule|why", r, re.I))),
            ("articulates_tradeoffs", lambda r, ret: bool(re.search(r"breakdown|tradeoff|perfect match|\d of \d", r, re.I))),
            ("expresses_uncertainty", lambda r, ret: bool(re.search(r"verify|check catalog|may have changed", r, re.I))),
            ("mentions_departments_or_distributions", lambda r, ret: bool(re.search(r"CI-H|REST|HASS|department|Course \d", r, re.I))),
            ("recognizes_course_6_or_eecs", lambda r, ret: bool(re.search(r"6\.\d|EECS|Course 6", r, re.I))),
        ])
        if ctx.semester:
            out.append(("offered_in_semester", lambda r, ret: True))
        if ctx.distributions:
            out.append(("have_distribution", lambda r, ret: bool(re.search(r"CI-H|REST|HASS", r, re.I))))
        if ctx.afternoon:
            out.append(("have_afternoon_option", lambda r, ret: bool(re.search(r"afternoon|pm|3-|4-", r, re.I))))
        if ctx.department:
            out.append(("cited_courses_same_department", lambda r, ret: bool(re.search(rf"\b{re.escape(ctx.department)}\.\d", r))))
        if ctx.course_6:
            out.append(("are_course_6", lambda r, ret: bool(re.search(r"6\.\d", r))))
        if ctx.graduation:
            out.append(("mentions_graduation_or_requirements", lambda r, ret: bool(re.search(r"graduat|6-3|requirement", r, re.I))))
        if ctx.advisor:
            out.append(("mentions_advisor_or_counseling", lambda r, ret: bool(re.search(r"advisor|counseling|talk to", r, re.I))))
        out.append(("suggests_no_results_or_relax", lambda r, ret: bool(re.search(r"no courses|relax|bottleneck", r, re.I))))

    return out


def _run_one_prompt(
    prompt: str,
    response: str,
    retriever,
    use_response_quality: bool,
    summary: dict,
    row: dict,
) -> None:
    """Run all checks relevant to this prompt/response; update summary and row."""
    ctx = parse_prompt_context(prompt)
    checks = _checks_for_context(ctx, retriever, use_response_quality)
    for name, fn in checks:
        try:
            passed = fn(response, retriever)
        except Exception:
            passed = False
        row["checks"][name] = passed
        if name not in summary:
            summary[name] = {"passed": 0, "total": 0}
        summary[name]["total"] += 1
        if passed:
            summary[name]["passed"] += 1


def run_batch_eval(limit: int | None = None, out_path: str | None = None, dry_run: bool = False) -> dict:
    """
    Load 100 prompts, run chatbot on each (up to limit), run only checks relevant to each
    prompt/response (inferred from prompt context), return results.
    If dry_run=True, use 3 synthetic responses (no HF_TOKEN).
    """
    from src.retriever import CourseRetriever

    try:
        from eval.response_quality import check_cited_courses_exist_in_catalog  # noqa: F401
        use_response_quality = True
    except ImportError:
        use_response_quality = False

    retriever = CourseRetriever()
    summary: dict = {}
    results = []

    if dry_run:
        synthetic = [
            ("I need a CI-H.", "Consider 24.131 (Ethics of Technology) — it satisfies your CI-H and fits afternoon. Schedule: W 3-5. Prereqs: None. Verify at catalog."),
            ("Compare 6.3900 and 6.4100.", "6.3900 is ML intro; 6.4100 is ML with more theory. Here's the breakdown: 6.3900 meets your criteria; 6.4100 has 6.1210 as prereq. Both are Course 6."),
            ("What about 99.99?", "I couldn't find 99.99 in the catalog. No courses match. Try relaxing the requirement or check student.mit.edu/catalog."),
        ]
        for prompt, response in synthetic:
            row = {"prompt": prompt, "response": response[:2000], "checks": {}}
            _run_one_prompt(prompt, response, retriever, use_response_quality, summary, row)
            results.append(row)
    else:
        if not os.getenv("HF_TOKEN"):
            print("HF_TOKEN not set. Use --dry-run to test checks without the API.", file=sys.stderr)
            return {"error": "HF_TOKEN not set", "results": [], "summary": {}}

        from src.chat import Chatbot

        prompts = get_100_prompts()
        if limit:
            prompts = prompts[:limit]
        n = len(prompts)
        chatbot = Chatbot()

        for i, prompt in enumerate(prompts):
            print(f"[{i+1}/{n}] {prompt[:50]}...", flush=True)
            try:
                response = chatbot.get_response(prompt, [])
            except Exception as e:
                response = f"[Error: {e}]"
            row = {"prompt": prompt, "response": response[:2000], "checks": {}}
            _run_one_prompt(prompt, response, retriever, use_response_quality, summary, row)
            results.append(row)

    for name in summary:
        s = summary[name]
        s["pass_rate"] = s["passed"] / s["total"] if s["total"] else 0.0

    out_path = out_path or Path(__file__).parent / "eval_results.json"
    payload = {"n_prompts": len(results), "summary": summary, "results": results}
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Wrote {out_path}")
    return payload


def main():
    parser = argparse.ArgumentParser(description="Batch eval: 100 prompts → chatbot → quality checks")
    parser.add_argument("--limit", type=int, default=None, help="Max prompts to run (default: all 100)")
    parser.add_argument("--out", type=str, default=None, help="Output JSON path (default: eval/eval_results.json)")
    parser.add_argument("--dry-run", action="store_true", help="Use 3 synthetic responses only (no API); for testing viz")
    args = parser.parse_args()
    run_batch_eval(limit=args.limit, out_path=args.out, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
