"""Re-test the retrieval comparisons with paired tests.

The main evaluation and the exploratory scripts (retrieval_ablation.py,
embedding_comparison.py, hybrid_rrf.py) compared configurations with
statsmodels' pairwise_tukeyhsd, which treats each configuration's per-question
scores as independent samples. All configurations answer the same questions,
so this module re-runs every comparison with a paired randomization test
(Holm-adjusted within each experiment and metric) and reports Tukey HSD
alongside for reference.

Output: results/paired_retrieval_stats.csv
"""

import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.stats import compare_configs

ALPHA = 0.01
METRICS = ["ndcg5", "recall5", "hit5", "mrr"]

MAIN_RENAMES = {"ndcg@5": "ndcg5", "recall@5": "recall5", "hit_rate@5": "hit5", "mrr": "mrr"}
MAIN_CONFIGS = ["bm25-k5-settlein_v4", "dense-k5-settlein_v4"]

# experiment -> (csv, config column, metric column renames, configs to keep or None for all)
# "main" is the planned BM25-vs-dense comparison; later exploratory runs in
# runs/ are tested in a separate family so they do not dilute its Holm correction.
EXPERIMENTS = {
    "main": ("retrieval_metrics.csv", "run", MAIN_RENAMES, MAIN_CONFIGS),
    "main_vs_exploratory": ("retrieval_metrics.csv", "run", MAIN_RENAMES, None),
    "retrieval_ablation": ("retrieval_ablation.csv", "config", {}, None),
    "embedding_comparison": ("embedding_comparison.csv", "config", {}, None),
    "hybrid_rrf": ("hybrid_rrf.csv", "config", {}, None),
    "rerank": ("rerank_experiment.csv", "config", {}, None),
}


def tukey_p(per_q: dict[str, dict[str, float]], g1: str, g2: str) -> float | None:
    """Unpaired Tukey HSD p-value for g1 vs g2 within the full family."""
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    data, labels = [], []
    for cfg, scores in per_q.items():
        data.extend(scores.values())
        labels.extend([cfg] * len(scores))
    res = pairwise_tukeyhsd(data, labels, alpha=ALPHA)
    for row in res.summary().data[1:]:
        if {row[0], row[1]} == {g1, g2}:
            return float(row[3])
    return None


def main():
    out_rows = []
    for exp, (fname, cfg_col, renames, keep) in EXPERIMENTS.items():
        path = REPO_ROOT / "results" / fname
        if not path.exists():
            print(f"Skipping {exp}: {fname} not found")
            continue
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        for r in rows:
            for old, new in renames.items():
                r[new] = r[old]
        rows = [r for r in rows if r.get("question_type") in ("known", "inferred")
                and (keep is None or r[cfg_col] in keep)]
        if len({r[cfg_col] for r in rows}) < 2:
            print(f"Skipping {exp}: fewer than two configs")
            continue

        print(f"\n== {exp} ({fname}) ==")
        for m in METRICS:
            per_q = {}
            for r in rows:
                per_q.setdefault(r[cfg_col], {})[r["question_id"]] = float(r[m])
            for c in compare_configs(per_q, m, alpha=ALPHA):
                c["experiment"] = exp
                c["p_tukey_unpaired"] = round(tukey_p(per_q, c["group1"], c["group2"]), 4)
                out_rows.append(c)
                flag = " *" if c["significant"] else ""
                print(f"  {m:<8} {c['group1']:>24} - {c['group2']:<24} diff={c['diff']:+.4f} "
                      f"p_paired={c['p']:.4f} p_holm={c['p_holm']:.4f} "
                      f"p_tukey={c['p_tukey_unpaired']:.4f}{flag}")

    fields = ["experiment"] + [k for k in out_rows[0] if k != "experiment"]
    out = REPO_ROOT / "results" / "paired_retrieval_stats.csv"
    with open(out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nWritten: {out}")


if __name__ == "__main__":
    main()
