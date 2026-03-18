"""
Load eval_results.json and plot pass rates per check.
Saves eval/eval_pass_rates.png and optionally eval/passes_per_prompt.png.

Usage: python -m eval.visualize_eval [--results path] [--out-dir dir]
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_results(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def plot_pass_rates(summary: dict, out_path: str) -> None:
    """Bar chart: pass rate (0–100%) per check."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. pip install matplotlib", file=sys.stderr)
        return

    names = list(summary.keys())
    rates = [summary[n]["pass_rate"] * 100 for n in names]
    colors = ["#2ecc71" if r >= 70 else "#f39c12" if r >= 40 else "#e74c3c" for r in rates]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(names, rates, color=colors)
    ax.set_xlim(0, 105)
    ax.set_xlabel("Pass rate (%)")
    ax.set_ylabel("Check")
    ax.set_title("Response quality: pass rate across 100 prompts")
    ax.axvline(70, color="gray", linestyle="--", alpha=0.7, label="70%")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def plot_passes_per_prompt(results: list[dict], out_path: str) -> None:
    """Histogram: number of checks passed per prompt (0 to n_checks)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if not results or "checks" not in results[0]:
        return
    n_checks = len(results[0]["checks"])
    counts = [sum(1 for v in r["checks"].values() if v) for r in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(counts, bins=range(0, n_checks + 2), align="left", rwidth=0.8, color="#3498db", edgecolor="white")
    ax.set_xlabel("Number of checks passed")
    ax.set_ylabel("Number of prompts")
    ax.set_title("Distribution of passes per prompt (100 prompts)")
    ax.set_xticks(range(0, n_checks + 1))
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize batch eval pass rates")
    parser.add_argument("--results", type=str, default=None, help="Path to eval_results.json")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory for output PNGs")
    args = parser.parse_args()

    eval_dir = Path(__file__).parent
    results_path = Path(args.results) if args.results else eval_dir / "eval_results.json"
    out_dir = Path(args.out_dir) if args.out_dir else eval_dir

    if not results_path.exists():
        print(f"Results not found: {results_path}. Run: python -m eval.run_batch_eval", file=sys.stderr)
        sys.exit(1)

    data = load_results(results_path)
    if "error" in data:
        print("No results (e.g. HF_TOKEN not set). Run eval with HF_TOKEN.", file=sys.stderr)
        sys.exit(1)

    summary = data.get("summary", {})
    results = data.get("results", [])

    if not summary:
        print("No summary in results file.", file=sys.stderr)
        sys.exit(1)

    # Save CSV summary for use without matplotlib
    csv_path = out_dir / "eval_pass_rates.csv"
    with open(csv_path, "w") as f:
        f.write("check,passed,total,pass_rate_pct\n")
        for name, s in summary.items():
            pr = s.get("pass_rate", 0) * 100
            f.write(f"{name},{s.get('passed', 0)},{s.get('total', 0)},{pr:.1f}\n")
    print(f"Saved {csv_path}")

    try:
        plot_pass_rates(summary, str(out_dir / "eval_pass_rates.png"))
        plot_passes_per_prompt(results, str(out_dir / "eval_passes_per_prompt.png"))
    except Exception as e:
        print(f"Plotting failed (install matplotlib?): {e}", file=sys.stderr)

    # Print table to stdout
    print("\nPass rates by check:")
    for name, s in summary.items():
        pr = s.get("pass_rate", 0) * 100
        print(f"  {name}: {s.get('passed', 0)}/{s.get('total', 0)} ({pr:.1f}%)")


if __name__ == "__main__":
    main()
