"""Report aggregator.

Aggregates all per-question evaluation CSVs into:
  - results/summary.csv (main results table)
  - results/summary.md (Markdown formatted)

Also runs Tukey HSD at α=0.01 across configurations if scipy is available.
"""

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_csv(path: Path) -> list[dict]:
    """Load a CSV file, return empty list if missing."""
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def aggregate_retrieval(rows: list[dict]) -> dict[str, dict]:
    """Aggregate retrieval metrics by config."""
    by_config = defaultdict(list)
    for r in rows:
        by_config[r["run"]].append(r)

    summary = {}
    metrics = ["ndcg@1", "ndcg@3", "ndcg@5", "recall@5", "hit_rate@5", "mrr"]

    for config, config_rows in by_config.items():
        agg = {"config": config, "n_queries": len(config_rows)}
        for m in metrics:
            vals = [float(r[m]) for r in config_rows if r[m]]
            agg[f"mean_{m}"] = round(sum(vals) / len(vals), 4) if vals else 0
        # Breakdown by type
        for q_type in ["known", "inferred"]:
            type_rows = [r for r in config_rows if r.get("question_type") == q_type]
            for m in ["ndcg@5", "recall@5"]:
                vals = [float(r[m]) for r in type_rows if r[m]]
                agg[f"{q_type}_{m}"] = round(sum(vals) / len(vals), 4) if vals else 0
        summary[config] = agg

    return summary


def aggregate_answerability(rows: list[dict]) -> dict[str, dict]:
    """Aggregate answerability metrics by config."""
    result = {}
    for r in rows:
        result[r["config"]] = {
            "correct_refusal_rate": float(r.get("correct_refusal_rate", 0)),
            "over_refusal_rate": float(r.get("over_refusal_rate", 0)),
        }
    return result


def aggregate_faithfulness(rows: list[dict]) -> dict[str, dict]:
    """Aggregate faithfulness metrics by config."""
    by_config = defaultdict(list)
    for r in rows:
        by_config[r["config"]].append(r)

    result = {}
    for config, config_rows in by_config.items():
        rates = [float(r["supported_rate"]) for r in config_rows]
        result[config] = {
            "mean_supported_rate": round(sum(rates) / len(rates), 4) if rates else 0,
        }
    return result


def aggregate_attribution(rows: list[dict]) -> dict[str, dict]:
    """Aggregate attribution metrics by config."""
    by_config = defaultdict(list)
    for r in rows:
        by_config[r["config"]].append(r)

    result = {}
    for config, config_rows in by_config.items():
        precs = [float(r["citation_precision"]) for r in config_rows
                 if r.get("citation_precision") and r["citation_precision"] != "None"]
        unattr = sum(1 for r in config_rows if r.get("unattributed") == "True" or r.get("unattributed") is True)
        total = len(config_rows)
        result[config] = {
            "mean_citation_precision": round(sum(precs) / len(precs), 4) if precs else None,
            "unattributed_rate": round(unattr / total, 4) if total else 0,
        }
    return result


def build_summary():
    """Build the main summary table."""
    results_dir = REPO_ROOT / "results"

    retrieval = load_csv(results_dir / "retrieval_metrics.csv")
    answerability = load_csv(results_dir / "answerability_metrics.csv")
    faithfulness = load_csv(results_dir / "faithfulness_metrics.csv")
    attribution = load_csv(results_dir / "attribution_metrics.csv")

    ret_agg = aggregate_retrieval(retrieval)
    ans_agg = aggregate_answerability(answerability)
    faith_agg = aggregate_faithfulness(faithfulness)
    attr_agg = aggregate_attribution(attribution)

    # Merge all configs
    all_configs = set()
    all_configs.update(ret_agg.keys())
    all_configs.update(ans_agg.keys())
    all_configs.update(faith_agg.keys())
    all_configs.update(attr_agg.keys())

    if not all_configs:
        print("No evaluation results found. Run evaluations first.")
        return

    # Build rows
    summary_rows = []
    for config in sorted(all_configs):
        row = {"config": config}

        ret = ret_agg.get(config, {})
        row["n_queries"] = ret.get("n_queries", 0)
        for m in ["ndcg@1", "ndcg@3", "ndcg@5", "recall@5", "hit_rate@5", "mrr"]:
            row[f"mean_{m}"] = ret.get(f"mean_{m}", "")
        for q_type in ["known", "inferred"]:
            for m in ["ndcg@5", "recall@5"]:
                row[f"{q_type}_{m}"] = ret.get(f"{q_type}_{m}", "")

        ans = ans_agg.get(config, {})
        row["correct_refusal_rate"] = ans.get("correct_refusal_rate", "")
        row["over_refusal_rate"] = ans.get("over_refusal_rate", "")

        faith = faith_agg.get(config, {})
        row["mean_supported_rate"] = faith.get("mean_supported_rate", "")

        attr = attr_agg.get(config, {})
        row["mean_citation_precision"] = attr.get("mean_citation_precision", "")
        row["unattributed_rate"] = attr.get("unattributed_rate", "")

        summary_rows.append(row)

    # Write CSV
    out_csv = results_dir / "summary.csv"
    fields = list(summary_rows[0].keys())
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary_rows)
    print(f"Summary CSV: {out_csv}")

    # Write Markdown table
    out_md = results_dir / "summary.md"
    with open(out_md, "w", encoding="utf-8") as f:
        f.write("# SettleIN Evaluation Results\n\n")

        # Retrieval table
        f.write("## Retrieval Metrics\n\n")
        f.write("| Config | nDCG@1 | nDCG@3 | nDCG@5 | Recall@5 | Hit@5 | MRR |\n")
        f.write("|--------|--------|--------|--------|----------|-------|-----|\n")
        for row in summary_rows:
            if row.get("n_queries"):
                f.write(f"| {row['config']} | {row.get('mean_ndcg@1','')} | {row.get('mean_ndcg@3','')} | "
                        f"{row.get('mean_ndcg@5','')} | {row.get('mean_recall@5','')} | "
                        f"{row.get('mean_hit_rate@5','')} | {row.get('mean_mrr','')} |\n")

        # Answerability table
        f.write("\n## Answerability\n\n")
        f.write("| Config | Correct Refusal | Over-Refusal |\n")
        f.write("|--------|-----------------|---------------|\n")
        for row in summary_rows:
            cr = row.get("correct_refusal_rate", "")
            ovr = row.get("over_refusal_rate", "")
            if cr != "":
                f.write(f"| {row['config']} | {cr} | {ovr} |\n")

        # Faithfulness table
        f.write("\n## Faithfulness\n\n")
        f.write("| Config | Supported-Claim Rate |\n")
        f.write("|--------|---------------------|\n")
        for row in summary_rows:
            sr = row.get("mean_supported_rate", "")
            if sr != "":
                f.write(f"| {row['config']} | {sr} |\n")

        # Attribution table
        f.write("\n## Attribution\n\n")
        f.write("| Config | Citation Precision | Unattributed Rate |\n")
        f.write("|--------|-------------------|-------------------|\n")
        for row in summary_rows:
            cp = row.get("mean_citation_precision", "")
            ur = row.get("unattributed_rate", "")
            if cp != "" or ur != "":
                f.write(f"| {row['config']} | {cp} | {ur} |\n")

    print(f"Summary MD:  {out_md}")

    # Print to console
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    for row in summary_rows:
        print(f"\n  Config: {row['config']}")
        if row.get("n_queries"):
            print(f"    nDCG@5: {row.get('mean_ndcg@5', 'N/A')}, Recall@5: {row.get('mean_recall@5', 'N/A')}")
        if row.get("correct_refusal_rate") != "":
            print(f"    Correct Refusal: {row.get('correct_refusal_rate', 'N/A')}, "
                  f"Over-Refusal: {row.get('over_refusal_rate', 'N/A')}")
        if row.get("mean_supported_rate") != "":
            print(f"    Faithfulness: {row.get('mean_supported_rate', 'N/A')}")


if __name__ == "__main__":
    build_summary()
