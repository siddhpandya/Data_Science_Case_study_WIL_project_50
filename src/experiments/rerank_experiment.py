"""Cross-encoder reranking experiment (exploratory: run after the main evaluation).

Configurations (all over the 107-passage index, S18 superseded excluded):
  bm25, dense, hybrid_rrf     original-text baselines (as in hybrid_rrf.py)
  hybrid_ctx                  RRF of contextual-header BM25 + contextual dense
  hybrid_ctx+rerank_base      hybrid_ctx top-20 reranked by bge-reranker-base
  hybrid_ctx+rerank_m3        hybrid_ctx top-20 reranked by bge-reranker-v2-m3

Retrieval metrics on the 35 answerable questions, with paired randomization
tests (Holm-adjusted per metric) and bootstrap 95% CIs.

Answerability signal: on all 48 questions, the top-1 score of each config is
used to separate answerable from out-of-KB questions (ROC AUC), and a
threshold sweep shows the correct-refusal / over-refusal trade-off.

Outputs:
  results/rerank_experiment.csv          per-question retrieval metrics
  results/rerank_stats.csv               pairwise paired tests
  results/rerank_answerability.csv       AUC + threshold sweep
  results/rerank_top1_scores.csv         top-1 score per question per config
  runs/rerank/{config}.txt               TREC runs (top 5)
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

from src.retrieval.search import load_collection, filter_superseded, BM25Index, DenseIndex
from src.retrieval.rerank import CrossEncoderReranker
from src.experiments.retrieval_ablation import (
    load_topics, load_qrels, ContextualBM25Index, ContextualDenseIndex,
    ndcg_at_k, recall_at_k, hit_at_k, mrr,
)
from src.experiments.hybrid_rrf import rrf_fuse
from src.evaluation.stats import compare_configs, bootstrap_ci, roc_auc, wilson_ci

POOL_DEPTH = 20
TOP_K = 5
RERANKERS = {
    "rerank_base": "BAAI/bge-reranker-base",
    "rerank_m3": "BAAI/bge-reranker-v2-m3",
}
METRICS = ["ndcg1", "ndcg3", "ndcg5", "recall5", "hit5", "mrr", "recall20"]


def rankings_for_question(qtext, idx, rerankers):
    """Return {config: ranked [(passage, score)]} for one question."""
    n = len(idx["bm25"].passages)
    bm25 = idx["bm25"].search(qtext, top_k=n)
    dense = idx["dense"].search(qtext, top_k=n)
    bm25_ctx = idx["bm25_ctx"].search(qtext, top_k=n)
    dense_ctx = idx["dense_ctx"].search(qtext, top_k=n)
    hybrid_ctx = rrf_fuse(bm25_ctx, dense_ctx)

    out = {
        "bm25": bm25,
        "dense": dense,
        "hybrid_rrf": rrf_fuse(bm25, dense),
        "hybrid_ctx": hybrid_ctx,
    }
    for name, rr in rerankers.items():
        out[f"hybrid_ctx+{name}"] = rr.rerank(qtext, hybrid_ctx[:POOL_DEPTH])
    return out


def main():
    topics = [t for t in load_topics() if t.get("question_id", "").strip()]
    qrels = load_qrels()
    passages = filter_superseded(load_collection(), include_superseded=False)
    print(f"Index: {len(passages)} passages, {len(topics)} questions")

    idx = {
        "bm25": BM25Index(passages),
        "dense": DenseIndex(passages),
        "bm25_ctx": ContextualBM25Index(passages),
        "dense_ctx": ContextualDenseIndex(passages),
    }
    rerankers = {name: CrossEncoderReranker(model) for name, model in RERANKERS.items()}

    per_q_rows, top1_rows, run_lines = [], [], {}
    for t in topics:
        qid, qtext, qtype = t["question_id"], t["question"], t["question_type"]
        t0 = time.time()
        ranked = rankings_for_question(qtext, idx, rerankers)
        print(f"  {qid} ({time.time() - t0:.1f}s)", flush=True)
        qr = qrels.get(qid, {})
        for cfg, results in ranked.items():
            rids = [p["id"] for p, _ in results]
            top1_rows.append({"config": cfg, "question_id": qid, "question_type": qtype,
                              "top1_score": round(results[0][1], 6) if results else 0.0})
            for rank, (p, s) in enumerate(results[:TOP_K], 1):
                run_lines.setdefault(cfg, []).append(f"{qid} Q0 {p['id']} {rank} {s:.6f} {cfg}\n")
            if qtype == "out_of_kb":
                continue
            per_q_rows.append({
                "config": cfg, "question_id": qid, "question_type": qtype,
                "ndcg1": ndcg_at_k(rids, qr, 1), "ndcg3": ndcg_at_k(rids, qr, 3),
                "ndcg5": ndcg_at_k(rids, qr, 5), "recall5": recall_at_k(rids, qr, 5),
                "hit5": hit_at_k(rids, qr, 5), "mrr": mrr(rids, qr),
                "recall20": recall_at_k(rids, qr, 20),
                "retrieved_ids": ";".join(rids[:TOP_K]),
            })

    results_dir = REPO / "results"
    write_csv(results_dir / "rerank_experiment.csv", per_q_rows)
    write_csv(results_dir / "rerank_top1_scores.csv", top1_rows)
    run_dir = REPO / "runs" / "rerank"
    run_dir.mkdir(parents=True, exist_ok=True)
    for cfg, lines in run_lines.items():
        with open(run_dir / f"{cfg.replace('+', '_')}.txt", "w", encoding="utf-8") as f:
            f.writelines(lines)

    configs = list(dict.fromkeys(r["config"] for r in per_q_rows))

    # ── Summary with bootstrap CIs ──
    print(f"\n{'Config':<26}" + "".join(f"{m:>18}" for m in METRICS))
    for cfg in configs:
        line = f"{cfg:<26}"
        for m in METRICS:
            vals = [r[m] for r in per_q_rows if r["config"] == cfg]
            lo, hi = bootstrap_ci(vals)
            line += f"  {np.mean(vals):.3f} [{lo:.2f},{hi:.2f}]"
        print(line)

    # ── Paired tests: every config against every other, one family per metric ──
    stats_rows = []
    for m in ["ndcg5", "recall5", "hit5", "mrr"]:
        per_q = {cfg: {r["question_id"]: r[m] for r in per_q_rows if r["config"] == cfg}
                 for cfg in configs}
        stats_rows.extend(compare_configs(per_q, m))
    write_csv(results_dir / "rerank_stats.csv", stats_rows)
    print("\nPaired randomization tests vs dense baseline (Holm-adjusted, alpha=0.01):")
    for r in stats_rows:
        if "dense" in (r["group1"], r["group2"]) and r["group1"] != "bm25":
            print(f"  {r['metric']:<8} {r['group1']} - {r['group2']}: diff={r['diff']:+.4f} "
                  f"CI[{r['diff_ci95_lo']:+.3f},{r['diff_ci95_hi']:+.3f}] "
                  f"W/L/T={r['wins']}/{r['losses']}/{r['ties']} "
                  f"p={r['p']:.4f} p_holm={r['p_holm']:.4f}{' *' if r['significant'] else ''}")

    # ── Answerability: can the top-1 score separate answerable from out-of-KB? ──
    ans_rows = []
    print("\nAnswerability signal (top-1 score, answerable vs out-of-KB):")
    for cfg in configs:
        rows = [r for r in top1_rows if r["config"] == cfg]
        pos = [r["top1_score"] for r in rows if r["question_type"] != "out_of_kb"]
        neg = [r["top1_score"] for r in rows if r["question_type"] == "out_of_kb"]
        auc = roc_auc(pos, neg)
        print(f"  {cfg:<26} AUC = {auc:.3f}")
        ans_rows.append({"config": cfg, "threshold": "AUC", "auc": round(auc, 4)})
        if "rerank" not in cfg:
            continue
        for thr in np.round(np.arange(0.05, 1.0, 0.05), 2):
            refused_ook = sum(s < thr for s in neg)
            refused_ans = sum(s < thr for s in pos)
            cr_lo, cr_hi = wilson_ci(refused_ook, len(neg))
            ans_rows.append({
                "config": cfg, "threshold": thr, "auc": round(auc, 4),
                "correct_refusal": round(refused_ook / len(neg), 4),
                "correct_refusal_ci95": f"[{cr_lo:.2f},{cr_hi:.2f}]",
                "over_refusal": round(refused_ans / len(pos), 4),
            })
    write_csv(results_dir / "rerank_answerability.csv", ans_rows)
    print(f"\nWritten: rerank_experiment.csv, rerank_stats.csv, rerank_answerability.csv, runs/rerank/")


def write_csv(path, rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
