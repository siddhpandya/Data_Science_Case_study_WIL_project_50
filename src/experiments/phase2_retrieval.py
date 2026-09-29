"""Phase 2: Run retrieval for all questions, generate TREC run files, then evaluate.

This script:
1. Runs BM25-k5 and Dense-k5 retrieval for all 48 questions
2. Writes TREC run files
3. Runs retrieval_eval on the 35 answerable questions
4. Reports fairness analysis on policy/student pairs
"""

import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.search import SearchEngine
from src.evaluation.retrieval_eval import evaluate_all_runs, load_topics


def write_trec_run(engine, topics, method, top_k, config_name):
    """Run retrieval and write TREC format run file."""
    runs_dir = REPO_ROOT / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    out_path = runs_dir / f"{config_name}.txt"

    results_map = {}
    
    with open(out_path, "w", encoding="utf-8") as f:
        for topic in topics:
            qid = topic["question_id"]
            question = topic["question"]
            
            results = engine.search(question, method=method, top_k=top_k)
            results_map[qid] = results
            
            for rank, (passage, score) in enumerate(results, 1):
                pid = passage["id"]
                f.write(f"{qid} Q0 {pid} {rank} {score:.6f} {config_name}\n")

    print(f"  Written: {out_path} ({sum(len(v) for v in results_map.values())} passages)")
    return results_map


def main():
    print("=" * 70)
    print("PHASE 2: Retrieval")
    print("=" * 70)

    # Load topics
    topics_path = REPO_ROOT / "data" / "topics.csv"
    with open(topics_path, "r", encoding="utf-8") as f:
        topics = [r for r in csv.DictReader(f) if r.get("question_id", "").strip()]
    
    print(f"Topics: {len(topics)}")
    answerable = [t for t in topics if t.get("question_type") != "out_of_kb"]
    out_of_kb = [t for t in topics if t.get("question_type") == "out_of_kb"]
    print(f"  Answerable: {len(answerable)} (known + inferred)")
    print(f"  Out-of-KB: {len(out_of_kb)}")

    # Load engine
    engine = SearchEngine()

    # BM25
    print("\nRunning BM25-k5...")
    bm25_results = write_trec_run(engine, topics, "bm25", 5, "bm25-k5-settlein_v4")

    # Dense
    print("\nRunning Dense-k5...")
    dense_results = write_trec_run(engine, topics, "dense", 5, "dense-k5-settlein_v4")

    # Evaluate retrieval (this excludes out_of_kb automatically)
    print("\n" + "=" * 70)
    print("RETRIEVAL METRICS (answerable questions only)")
    print("=" * 70)
    all_results = evaluate_all_runs()

    # Print per-question results
    if all_results:
        print("\n\nPer-question nDCG@5:")
        for r in all_results:
            mark = "✅" if r["hit_rate@5"] > 0 else "❌"
            print(f"  {mark} [{r['question_id']}] ({r['question_type']}) "
                  f"nDCG@5={r['ndcg@5']:.4f} Recall@5={r['recall@5']:.4f} "
                  f"relevant={r['num_relevant']} | run={r['run']}")

    return bm25_results, dense_results


if __name__ == "__main__":
    main()
