"""Answerability evaluation.

Two metrics:
  1. Correct refusal rate: % of out-of-KB questions that were refused (higher = better)
  2. Over-refusal rate: % of Known/Inferred questions that were refused (lower = better)

Breaks correct refusal rate down by refusal_mechanism (threshold vs prompt).

With only 13 out-of-KB questions one question moves the correct-refusal rate
by 7.7 points, so every rate is reported with a 95% Wilson interval. Rates are
reported for strict refusals (exact refusal string, the primary metric) and
lenient refusals (any decline phrase). Balanced accuracy averages correct
refusal and (1 - over-refusal) so the two error types weigh equally.

Output: results/answerability_metrics.csv
"""

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.stats import wilson_ci


def load_generations(config_name: str) -> list[dict]:
    """Load generation JSONL for a config."""
    gen_path = REPO_ROOT / "results" / "generations" / f"{config_name}.jsonl"
    records = []
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def evaluate_answerability(config_name: str) -> dict:
    """Compute answerability metrics for a single configuration."""
    generations = load_generations(config_name)

    out_of_kb = [g for g in generations if g["question_type"] == "out_of_kb"]
    answerable = [g for g in generations if g["question_type"] in ("known", "inferred")]

    # Correct refusal rate
    correct_refusals = sum(1 for g in out_of_kb if g["refused"])
    correct_refusal_rate = correct_refusals / len(out_of_kb) if out_of_kb else 0.0

    # Over-refusal rate
    over_refusals = sum(1 for g in answerable if g["refused"])
    over_refusal_rate = over_refusals / len(answerable) if answerable else 0.0

    # Breakdown by mechanism
    mechanism_counts = {}
    for g in out_of_kb:
        if g["refused"]:
            mech = g.get("refusal_mechanism", "unknown")
            mechanism_counts[mech] = mechanism_counts.get(mech, 0) + 1

    cr_lo, cr_hi = wilson_ci(correct_refusals, len(out_of_kb))
    or_lo, or_hi = wilson_ci(over_refusals, len(answerable))

    # Lenient: any decline phrase counts as a refusal
    lenient_cr = sum(1 for g in out_of_kb if g.get("refused_lenient", g["refused"]))
    lenient_or = sum(1 for g in answerable if g.get("refused_lenient", g["refused"]))

    result = {
        "config": config_name,
        "total_questions": len(generations),
        "out_of_kb_count": len(out_of_kb),
        "answerable_count": len(answerable),
        "correct_refusals": correct_refusals,
        "correct_refusal_rate": round(correct_refusal_rate, 4),
        "over_refusals": over_refusals,
        "over_refusal_rate": round(over_refusal_rate, 4),
        "correct_refusal_ci95": f"[{cr_lo:.3f}, {cr_hi:.3f}]",
        "over_refusal_ci95": f"[{or_lo:.3f}, {or_hi:.3f}]",
        "balanced_accuracy": round((correct_refusal_rate + 1 - over_refusal_rate) / 2, 4),
        "correct_refusal_rate_lenient": round(lenient_cr / len(out_of_kb), 4) if out_of_kb else 0.0,
        "over_refusal_rate_lenient": round(lenient_or / len(answerable), 4) if answerable else 0.0,
        "refusal_by_mechanism": json.dumps(mechanism_counts),
    }

    return result


def evaluate_all_configs() -> list[dict]:
    """Evaluate answerability for all generation configs."""
    gen_dir = REPO_ROOT / "results" / "generations"
    results = []

    for gen_file in sorted(gen_dir.glob("*.jsonl")):
        config_name = gen_file.stem
        # Skip currency-only generation files (they don't have question_type)
        if "currency" in config_name:
            print(f"Skipping answerability: {config_name} (currency subset)")
            continue
        print(f"Evaluating answerability: {config_name}")
        result = evaluate_answerability(config_name)
        results.append(result)
        print(f"  Correct refusal rate: {result['correct_refusal_rate']:.1%} {result['correct_refusal_ci95']}")
        print(f"  Over-refusal rate:    {result['over_refusal_rate']:.1%} {result['over_refusal_ci95']}")
        print(f"  Balanced accuracy:    {result['balanced_accuracy']:.1%}")

    if results:
        out_path = REPO_ROOT / "results" / "answerability_metrics.csv"
        fields = list(results[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nWritten: {out_path}")

    return results


if __name__ == "__main__":
    evaluate_all_configs()
