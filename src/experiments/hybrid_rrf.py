"""Hybrid search by Reciprocal Rank Fusion (RRF).

Fuses BM25 and dense (bge-small-en-v1.5) rankings using RRF:
    score(p) = sum of 1 / (k + rank) for each ranking, k = 60.

Retrieval-only experiment: no generation.
"""

import csv
import json
import math
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO = Path(r"c:\Users\Abhishek\OneDrive\Documents\Data_Science_Case_study_WIL_project_50")
sys.path.insert(0, str(REPO))

from src.retrieval.search import (
    load_collection, filter_superseded,
    BM25Index, DenseIndex,
)


# ── Load data ────────────────────────────────────────────────────────────────

def load_topics():
    with open(REPO / "data" / "topics.csv", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def load_qrels():
    qrels = {}
    with open(REPO / "data" / "qrels.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, pid, grade = parts[:4]
                qrels.setdefault(qid, {})[pid] = int(grade)
    return qrels


# ── Metrics ──────────────────────────────────────────────────────────────────

def ndcg_at_k(retrieved_ids, qrels_for_q, k):
    dcg = 0.0
    for i, pid in enumerate(retrieved_ids[:k]):
        rel = qrels_for_q.get(pid, 0)
        dcg += rel / math.log2(i + 2)
    ideal = sorted(qrels_for_q.values(), reverse=True)[:k]
    idcg = sum(r / math.log2(i + 2) for i, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0

def recall_at_k(retrieved_ids, qrels_for_q, k):
    relevant = {pid for pid, g in qrels_for_q.items() if g > 0}
    if not relevant:
        return 0.0
    return len(relevant & set(retrieved_ids[:k])) / len(relevant)

def hit_at_k(retrieved_ids, qrels_for_q, k):
    relevant = {pid for pid, g in qrels_for_q.items() if g > 0}
    return 1.0 if relevant & set(retrieved_ids[:k]) else 0.0

def mrr(retrieved_ids, qrels_for_q):
    for i, pid in enumerate(retrieved_ids):
        if qrels_for_q.get(pid, 0) > 0:
            return 1.0 / (i + 1)
    return 0.0


# ── RRF Fusion ───────────────────────────────────────────────────────────────

RRF_K = 60  # Standard RRF constant

def rrf_fuse(bm25_ranking, dense_ranking):
    """Fuse two rankings using Reciprocal Rank Fusion.

    Each ranking is a list of (passage_dict, score) sorted by score descending.
    Returns a list of (passage_dict, rrf_score) sorted by rrf_score descending.
    """
    scores = {}      # pid -> rrf_score
    passages = {}    # pid -> passage_dict

    for rank, (passage, _) in enumerate(bm25_ranking, 1):
        pid = passage["id"]
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_K + rank)
        passages[pid] = passage

    for rank, (passage, _) in enumerate(dense_ranking, 1):
        pid = passage["id"]
        scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_K + rank)
        passages[pid] = passage

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(passages[pid], score) for pid, score in ranked]


# ── Evaluation ───────────────────────────────────────────────────────────────

def evaluate_all(bm25_idx, dense_idx, questions, qrels, top_k=5):
    """Evaluate BM25, dense, and hybrid-RRF. Returns per-question results."""
    all_results = []

    for q in questions:
        qid = q["question_id"]
        qtext = q["question"]
        qtype = q["question_type"]

        if qtype == "out_of_kb":
            continue

        qr = qrels.get(qid, {})

        # BM25 full ranking (all docs)
        bm25_results = bm25_idx.search(qtext, top_k=len(bm25_idx.passages))
        bm25_rids = [p["id"] for p, _ in bm25_results]

        # Dense full ranking (all docs)
        dense_results = dense_idx.search(qtext, top_k=len(dense_idx.passages))
        dense_rids = [p["id"] for p, _ in dense_results]

        # RRF fusion using full ranked lists
        rrf_results = rrf_fuse(bm25_results, dense_results)
        rrf_rids = [p["id"] for p, _ in rrf_results]

        for name, rids in [("bm25", bm25_rids), ("dense", dense_rids), ("hybrid_rrf", rrf_rids)]:
            all_results.append({
                "config": name,
                "question_id": qid,
                "question_type": qtype,
                "ndcg1": ndcg_at_k(rids, qr, 1),
                "ndcg3": ndcg_at_k(rids, qr, 3),
                "ndcg5": ndcg_at_k(rids, qr, 5),
                "recall5": recall_at_k(rids, qr, 5),
                "hit5": hit_at_k(rids, qr, 5),
                "mrr": mrr(rids, qr),
                "retrieved_ids": rids[:5],
            })

    return all_results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    passages = load_collection()
    passages = filter_superseded(passages, include_superseded=False)
    print(f"Passages: {len(passages)}")

    topics = load_topics()
    qrels = load_qrels()
    answerable = [t for t in topics if t["question_type"] != "out_of_kb"]
    print(f"Answerable questions: {len(answerable)}")

    # Build indices
    print("\n=== Building indices ===")
    bm25_idx = BM25Index(passages)
    print(f"  BM25 index: {bm25_idx.n_docs} docs")
    dense_idx = DenseIndex(passages)
    dense_idx._build_embeddings()
    print(f"  Dense index: {dense_idx._embeddings.shape}")

    # Evaluate
    print("\n=== Evaluating ===")
    all_results = evaluate_all(bm25_idx, dense_idx, topics, qrels)

    # Save CSV
    out_path = REPO / "results" / "hybrid_rrf.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["config", "question_id", "question_type",
                  "ndcg1", "ndcg3", "ndcg5", "recall5", "hit5", "mrr", "retrieved_ids"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            row = dict(r)
            row["retrieved_ids"] = ";".join(row["retrieved_ids"])
            writer.writerow(row)
    print(f"\n  Results: {out_path}")

    # Write TREC run for hybrid
    run_dir = REPO / "runs" / "hybrid"
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / "hybrid_rrf.txt"
    with open(run_path, "w") as f:
        for q in topics:
            qid = q["question_id"]
            if q["question_type"] == "out_of_kb":
                continue
            bm25_r = bm25_idx.search(q["question"], top_k=len(passages))
            dense_r = dense_idx.search(q["question"], top_k=len(passages))
            rrf = rrf_fuse(bm25_r, dense_r)
            for rank, (p, score) in enumerate(rrf[:1000], 1):
                f.write(f"{qid} Q0 {p['id']} {rank} {score:.6f} hybrid_rrf\n")
    print(f"  TREC run: {run_path}")

    # ── Print results ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("HYBRID RRF RESULTS")
    print(f"{'='*80}")

    for config in ["bm25", "dense", "hybrid_rrf"]:
        for split_name, split_types in [("Known", {"known"}), ("Inferred", {"inferred"}),
                                         ("Combined", {"known", "inferred"})]:
            rows = [r for r in all_results
                    if r["config"] == config and r["question_type"] in split_types]
            if not rows:
                continue
            n = len(rows)
            means = {m: np.mean([r[m] for r in rows])
                     for m in ["ndcg1", "ndcg3", "ndcg5", "recall5", "hit5", "mrr"]}
            print(f"\n  {config} [{split_name}, n={n}]")
            print(f"    nDCG@1={means['ndcg1']:.4f}  nDCG@3={means['ndcg3']:.4f}  nDCG@5={means['ndcg5']:.4f}")
            print(f"    Recall@5={means['recall5']:.4f}  Hit@5={means['hit5']:.4f}  MRR={means['mrr']:.4f}")

    # ── Diagnostic questions ─────────────────────────────────────────────────
    diag_qids = ["T01Q01", "T03Q01", "T05Q01", "T11Q02", "T25Q02"]
    print(f"\n{'='*80}")
    print("DIAGNOSTIC QUESTIONS")
    print(f"{'='*80}")

    for qid in diag_qids:
        print(f"\n  {qid}:")
        for config in ["bm25", "dense", "hybrid_rrf"]:
            row = next((r for r in all_results
                       if r["config"] == config and r["question_id"] == qid), None)
            if row:
                rids = row["retrieved_ids"]
                print(f"    {config:15s}  nDCG@5={row['ndcg5']:.4f}  Hit@5={row['hit5']:.0f}"
                      f"  retrieved={rids}")

    # ── Fairness ─────────────────────────────────────────────────────────────
    # Fairness pairs: Q01 is policy, Q02 (variant_of=Q01) is student
    fairness_pairs = {}
    for t in topics:
        if t.get("variant_of", "").strip():
            tid = t["topic_id"]
            policy_qid = t["variant_of"].strip()
            student_qid = t["question_id"]
            fairness_pairs[tid] = (policy_qid, student_qid)

    print(f"\n{'='*80}")
    print("FAIRNESS: nDCG@5 for policy/student pairs")
    print(f"{'='*80}")

    for config in ["bm25", "dense", "hybrid_rrf"]:
        print(f"\n  {config}:")
        for tid in sorted(fairness_pairs.keys()):
            pol_qid, stu_qid = fairness_pairs[tid]
            pol_row = next((r for r in all_results
                           if r["config"] == config and r["question_id"] == pol_qid), None)
            stu_row = next((r for r in all_results
                           if r["config"] == config and r["question_id"] == stu_qid), None)
            if pol_row and stu_row:
                gap = abs(pol_row["ndcg5"] - stu_row["ndcg5"])
                print(f"    {tid}: policy={pol_row['ndcg5']:.4f}  student={stu_row['ndcg5']:.4f}  gap={gap:.4f}")

    # ── Tukey HSD ────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("TUKEY HSD (alpha = 0.01) — diff = group1 minus group2")
    print(f"{'='*80}")

    try:
        from statsmodels.stats.multicomp import pairwise_tukeyhsd

        models = ["bm25", "dense", "hybrid_rrf"]
        for metric in ["ndcg5", "recall5", "hit5", "mrr"]:
            groups = {}
            for m in models:
                groups[m] = [r[metric] for r in all_results if r["config"] == m]

            if all(len(v) > 0 for v in groups.values()):
                data = []
                labels = []
                for m in models:
                    data.extend(groups[m])
                    labels.extend([m] * len(groups[m]))

                tukey = pairwise_tukeyhsd(data, labels, alpha=0.01)
                print(f"\n  {metric}:")
                for row in tukey.summary().data[1:]:
                    g1, g2, meandiff, p_adj, lower, upper, reject = row
                    diff = -meandiff  # negate: first-named minus second-named
                    sig = "sig" if reject else "ns"
                    print(f"    {g1:15s} − {g2:15s}: diff={diff:+.4f}  p={p_adj:.4f}  ({sig})")
    except ImportError as e:
        print(f"  Could not run Tukey HSD: {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
