"""Fairness evaluation.

Uses paraphrase variants to measure retrieval consistency across
different phrasings (policy vs student register).

Metrics:
  - nDCG@5 by register (policy vs student)
  - Register gap: difference between best and worst variant performance
  - Mean within-topic standard deviation

Output: results/fairness_metrics.csv

NOTE: Requires topics.csv with `register` and `variant_of` fields populated.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def evaluate_fairness(run_name: str) -> list[dict]:
    """Evaluate fairness for a single run."""
    from src.evaluation.retrieval_eval import load_qrels, load_run, load_topics, ndcg_at_k

    runs_dir = REPO_ROOT / "runs"
    run_path = runs_dir / f"{run_name}.txt"
    if not run_path.exists():
        print(f"  Run file not found: {run_path}")
        return []

    qrels = load_qrels()
    run = load_run(run_path)
    topics = load_topics()

    # Group questions by topic_id
    topic_groups = defaultdict(list)
    for qid, topic in topics.items():
        tid = topic.get("topic_id", "")
        register = topic.get("register", "")
        if tid and register:  # Only include questions with register tags
            topic_groups[tid].append({
                "question_id": qid,
                "register": register,
                "topic_id": tid,
            })

    if not topic_groups:
        print("  No questions with register tags found")
        return []

    results = []
    for tid, questions in sorted(topic_groups.items()):
        ndcg_scores = {}
        for q in questions:
            qid = q["question_id"]
            ranked = run.get(qid, [])
            ranked_pids = [pid for pid, _ in ranked]
            qrel = qrels.get(qid, {})
            ranked_rels = [qrel.get(pid, 0) for pid in ranked_pids]
            all_rels = list(qrel.values())

            score = ndcg_at_k(ranked_rels, all_rels, 5)
            ndcg_scores[qid] = {
                "score": score,
                "register": q["register"],
            }

        # Compute metrics
        scores = [v["score"] for v in ndcg_scores.values()]
        policy_scores = [v["score"] for v in ndcg_scores.values() if v["register"] == "policy"]
        student_scores = [v["score"] for v in ndcg_scores.values() if v["register"] == "student"]

        import statistics
        std_dev = statistics.stdev(scores) if len(scores) > 1 else 0.0

        results.append({
            "topic_id": tid,
            "run": run_name,
            "num_variants": len(scores),
            "mean_ndcg5": round(sum(scores) / len(scores), 4) if scores else 0,
            "std_ndcg5": round(std_dev, 4),
            "policy_mean": round(sum(policy_scores) / len(policy_scores), 4) if policy_scores else None,
            "student_mean": round(sum(student_scores) / len(student_scores), 4) if student_scores else None,
            "register_gap": round(
                abs((sum(policy_scores) / len(policy_scores)) - (sum(student_scores) / len(student_scores))), 4
            ) if policy_scores and student_scores else None,
            "best_variant": round(max(scores), 4),
            "worst_variant": round(min(scores), 4),
        })

    return results


def evaluate_all_runs():
    """Evaluate fairness for all run files."""
    runs_dir = REPO_ROOT / "runs"
    all_results = []

    for run_path in sorted(runs_dir.glob("*.txt")):
        if run_path.name == ".gitkeep":
            continue
        run_name = run_path.stem
        print(f"Evaluating fairness: {run_name}")
        results = evaluate_fairness(run_name)
        all_results.extend(results)

    if all_results:
        out_path = REPO_ROOT / "results" / "fairness_metrics.csv"
        fields = list(all_results[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nWritten: {out_path}")
    else:
        print("\nNo fairness results (need questions with register=policy/student in topics.csv)")

    return all_results


if __name__ == "__main__":
    evaluate_all_runs()
