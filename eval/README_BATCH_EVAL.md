# Batch evaluation: 100 prompts and pass-rate visualization

This workflow generates **100 sample prompts**, runs the chatbot on each (when `HF_TOKEN` is set), evaluates every response with **7 universal response-quality checks**, and produces **pass-rate summaries and visualizations**.

## Steps

### 1. Generate 100 prompts

Prompts are defined in `eval/get_prompts_100.py` and cover:

- Distribution (CI-H, REST, HASS, LAB)
- Prerequisites and scheduling
- Course comparisons and exact-course lookups
- Program/degree (6-3, Course 8, TPP, etc.)
- Constraint bundles and edge cases

No separate generation step is required; the run script loads them automatically.

### 2. Run batch evaluation

From the **project root**:

```bash
# Full run (requires HF_TOKEN; ~100 LLM calls)
python -m eval.run_batch_eval

# Limit to first N prompts (e.g. 5 for a quick test)
python -m eval.run_batch_eval --limit 5

# Dry run: 3 synthetic responses only, no API (for testing)
python -m eval.run_batch_eval --dry-run
```

Output: **`eval/eval_results.json`** with:

- `n_prompts`: number of (prompt, response) pairs
- `summary`: for each check, `passed`, `total`, `pass_rate`
- `results`: per-prompt `prompt`, `response` (truncated), and `checks` (check name → bool)

### 3. Visualize pass rates

```bash
python -m eval.visualize_eval
# Or: python -m eval.visualize_eval --results eval/eval_results.json --out-dir eval
```

This:

- Writes **`eval/eval_pass_rates.csv`** (check, passed, total, pass_rate_pct) for use in spreadsheets or other tools.
- If **matplotlib** is installed, also produces:
  - **`eval/eval_pass_rates.png`** — horizontal bar chart of pass rate (%) per check (green ≥70%, orange ≥40%, red &lt;40%).
  - **`eval/eval_passes_per_prompt.png`** — histogram of how many checks each prompt passed (0 to n_checks).

Pass rates are also printed to stdout.

## Checks: only those relevant to each prompt

Checks are **context-dependent**: we parse each prompt (see `eval/prompt_context.py`) and run only checks that apply.

- **Always run:** `cited_courses_exist`, `not_overwhelming`, `explains_recommendations`, `articulates_tradeoffs`, `expresses_uncertainty`, `mentions_departments_or_distributions`, `suggests_no_results_or_relax`
- **Run when prompt mentions semester (Fall/Spring):** `offered_in_semester`
- **Run when prompt mentions a distribution (CI-H, REST, HASS, etc.):** `have_distribution`
- **Run when prompt mentions afternoon:** `have_afternoon_option`
- **Run when prompt mentions a department (Course 8, 6-3, etc.):** `cited_courses_same_department` (and, for Course 6, `are_course_6`, `recognizes_course_6_or_eecs`)
- **Run when prompt mentions HASS/CI-H:** `have_ci_h_or_hass`, `mentions_hass_or_cih`
- **Run when prompt mentions excluded prereqs (e.g. “don’t have 6.031”):** `prereqs_exclude`
- **Run when prompt mentions seminars/small class:** `cited_courses_small_class`
- **Run when prompt mentions graduation/6-3:** `mentions_graduation_or_requirements`
- **Run when prompt asks for advisor/counseling:** `mentions_advisor_or_counseling`
- **Run when prompt mentions 6.UAT:** `mentions_6_UAT_restrictions`
- **Run when prompt is vague/short:** `asks_clarifying_question`

So each response is evaluated with **only the checks that are relevant**; the summary reports pass rate per check over the subset of prompts where that check was run (e.g. `offered_in_semester` might have `total: 12` if only 12 prompts mentioned a semester).

When `eval/response_quality.py` is present, the script uses those implementations; otherwise it uses a minimal regex-based fallback so `--dry-run` and CSV/viz still work.

## Dependencies

- **Run (live):** `HF_TOKEN` in environment, `src.chat`, `src.retriever`, and (optionally) `eval.response_quality`
- **Visualization:** `matplotlib` for PNGs (optional); CSV and printed summary always run.  
  Add to `requirements.txt`: `matplotlib>=3.7.0`

## Example: report-ready pass rates

After a full run:

```bash
python -m eval.run_batch_eval
python -m eval.visualize_eval
```

Use **`eval/eval_pass_rates.csv`** or **`eval/eval_pass_rates.png`** in your project report. The CSV columns are: `check`, `passed`, `total`, `pass_rate_pct`.
