"""Retrieval evaluation: nDCG, Recall, Hit Rate, MRR.

Reads TREC run files from runs/ and qrels from data/qrels.txt.
Reports separately for Known and Inferred question types.

Output: results/retrieval_metrics.csv
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from math import log2

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_qrels() -> dict[str, dict[str, int]]:
    """Load qrels.txt → {question_id: {passage_id: relevance}}."""
    qrels_path = REPO_ROOT / "data" / "qrels.txt"
    qrels = defaultdict(dict)
    with open(qrels_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 4:
                parts = line.split()
            if len(parts) >= 4:
                qid, _, pid, rel = parts[0], parts[1], parts[2], int(parts[3])
                qrels[qid][pid] = rel
    return dict(qrels)


def load_run(run_path: Path) -> dict[str, list[tuple[str, float]]]:
    """Load TREC run file → {question_id: [(passage_id, score), ...]}."""
    run = defaultdict(list)
    with open(run_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 6:
                qid, _, pid, rank, score, tag = parts[:6]
                run[qid].append((pid, float(score)))
    # Sort by score descending
    for qid in run:
        run[qid].sort(key=lambda x: x[1], reverse=True)
    return dict(run)


def load_topics() -> dict[str, dict]:
    """Load topics.csv → {question_id: row}."""
    topics_path = REPO_ROOT / "data" / "topics.csv"
    topics = {}
    with open(topics_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qid = r.get("question_id", "").strip()
            if qid:
                topics[qid] = r
    return topics


def dcg_at_k(ranked_rels: list[int], k: int) -> float:
    """Compute DCG@k."""
    dcg = 0.0
    for i, rel in enumerate(ranked_rels[:k]):
        dcg += rel / log2(i + 2)  # i+2 because i is 0-indexed
    return dcg


def ndcg_at_k(ranked_rels: list[int], all_rels: list[int], k: int) -> float:
    """Compute nDCG@k."""
    dcg = dcg_at_k(ranked_rels, k)
    ideal = dcg_at_k(sorted(all_rels, reverse=True), k)
    if ideal == 0:
        return 0.0
    return dcg / ideal


def recall_at_k(ranked_pids: list[str], relevant_pids: set[str], k: int) -> float:
    """Fraction of relevant docs found in top-k."""
    if not relevant_pids:
        return 0.0
    found = len(set(ranked_pids[:k]) & relevant_pids)
    return found / len(relevant_pids)


def hit_rate_at_k(ranked_pids: list[str], relevant_pids: set[str], k: int) -> float:
    """1 if any relevant doc in top-k, else 0."""
    return 1.0 if set(ranked_pids[:k]) & relevant_pids else 0.0


def mrr(ranked_pids: list[str], relevant_pids: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant doc."""
    for i, pid in enumerate(ranked_pids):
        if pid in relevant_pids:
            return 1.0 / (i + 1)
    return 0.0


def evaluate_run(run_path: Path) -> list[dict]:
    """Evaluate a single run file against qrels."""
    qrels = load_qrels()
    run = load_run(run_path)
    topics = load_topics()

    results = []

    for qid in sorted(set(list(run.keys()) + list(qrels.keys()))):
        topic = topics.get(qid, {})
        q_type = topic.get("question_type", "unknown")

        # Skip out-of-kb questions (no qrels)
        if q_type == "out_of_kb":
            continue

        ranked = run.get(qid, [])
        ranked_pids = [pid for pid, _ in ranked]
        qrel = qrels.get(qid, {})
        relevant_pids = {pid for pid, rel in qrel.items() if rel > 0}
        all_rels = list(qrel.values())

        # Get relevance scores for ranked list
        ranked_rels = [qrel.get(pid, 0) for pid in ranked_pids]

        row = {
            "question_id": qid,
            "question_type": q_type,
            "run": run_path.stem,
            "ndcg@1": round(ndcg_at_k(ranked_rels, all_rels, 1), 4),
            "ndcg@3": round(ndcg_at_k(ranked_rels, all_rels, 3), 4),
            "ndcg@5": round(ndcg_at_k(ranked_rels, all_rels, 5), 4),
            "recall@5": round(recall_at_k(ranked_pids, relevant_pids, 5), 4),
            "hit_rate@5": round(hit_rate_at_k(ranked_pids, relevant_pids, 5), 4),
            "mrr": round(mrr(ranked_pids, relevant_pids), 4),
            "num_relevant": len(relevant_pids),
            "num_retrieved": len(ranked_pids),
        }
        results.append(row)

    return results


def evaluate_all_runs() -> list[dict]:
    """Evaluate all TREC run files in runs/."""
    runs_dir = REPO_ROOT / "runs"
    all_results = []

    run_files = sorted(runs_dir.glob("*.txt"))
    if not run_files:
        print("No run files found in runs/", file=sys.stderr)
        return []

    for run_path in run_files:
        if run_path.name == ".gitkeep":
            continue
        print(f"Evaluating: {run_path.name}")
        results = evaluate_run(run_path)
        all_results.extend(results)
        # Print summary
        if results:
            metrics = ["ndcg@1", "ndcg@3", "ndcg@5", "recall@5", "hit_rate@5", "mrr"]
            for m in metrics:
                vals = [r[m] for r in results]
                avg = sum(vals) / len(vals) if vals else 0
                print(f"  {m}: {avg:.4f}")

    # Write CSV
    if all_results:
        out_path = REPO_ROOT / "results" / "retrieval_metrics.csv"
        fields = list(all_results[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nWritten: {out_path}")

    return all_results


if __name__ == "__main__":
    evaluate_all_runs()
