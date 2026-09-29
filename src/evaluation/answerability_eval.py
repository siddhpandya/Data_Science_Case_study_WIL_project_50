"""Answerability evaluation.

Two metrics:
  1. Correct refusal rate: % of out-of-KB questions that were refused (higher = better)
  2. Over-refusal rate: % of Known/Inferred questions that were refused (lower = better)

Breaks correct refusal rate down by refusal_mechanism (threshold vs prompt).

Output: results/answerability_metrics.csv
"""

import csv
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


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

    result = {
        "config": config_name,
        "total_questions": len(generations),
        "out_of_kb_count": len(out_of_kb),
        "answerable_count": len(answerable),
        "correct_refusals": correct_refusals,
        "correct_refusal_rate": round(correct_refusal_rate, 4),
        "over_refusals": over_refusals,
        "over_refusal_rate": round(over_refusal_rate, 4),
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
        print(f"  Correct refusal rate: {result['correct_refusal_rate']:.1%}")
        print(f"  Over-refusal rate:    {result['over_refusal_rate']:.1%}")

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
