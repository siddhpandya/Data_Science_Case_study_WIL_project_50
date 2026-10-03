"""Currency evaluation.

For currency-sensitive questions:
  - Stale citation rate: fraction citing a superseded passage
  - Current preference rate: when both current and stale are retrievable,
    how often the current one ranks higher
  - Corpus freshness: % of passages with last_updated within 12 months (descriptive)

Output: results/currency_metrics.csv

NOTE: Requires `is_superseded` and `last_updated` fields in collection.jsonl,
and `currency_sensitive` field in topics.csv. Will produce empty results if
these fields are not populated.
"""

import csv
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_collection_meta() -> dict[str, dict]:
    """Load collection with metadata."""
    col_path = REPO_ROOT / "data" / "collection.jsonl"
    passages = {}
    with open(col_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line.strip())
                passages[p["id"]] = p
    return passages


def evaluate_currency(config_name: str) -> list[dict]:
    """Evaluate currency for a single configuration."""
    gen_path = REPO_ROOT / "results" / "generations" / f"{config_name}.jsonl"
    topics_path = REPO_ROOT / "data" / "topics.csv"

    generations = []
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                generations.append(json.loads(line.strip()))

    # Load topics to find currency-sensitive questions
    topics = {}
    with open(topics_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qid = r.get("question_id", "").strip()
            if qid:
                topics[qid] = r

    collection = load_collection_meta()
    # Superseded status is recorded per source in data/sources_draft.csv;
    # collection.jsonl has no is_superseded field.
    from src.retrieval.search import load_superseded_sources
    superseded_sources = load_superseded_sources()

    def is_stale(pid: str) -> bool:
        p = collection.get(pid, {})
        return bool(p.get("is_superseded", False)) or p.get("source_id") in superseded_sources

    # Filter to currency-sensitive questions
    currency_gens = [
        g for g in generations
        if topics.get(g["question_id"], {}).get("currency_sensitive", "").lower() == "true"
    ]

    if not currency_gens:
        print(f"  No currency-sensitive questions found for {config_name}")
        return []

    results = []
    for gen in currency_gens:
        retrieved_ids = gen.get("retrieved_ids", [])
        cited_ids = gen.get("cited_ids", [])

        # Check if any cited/retrieved passages are superseded
        total_cited = len(cited_ids)
        stale_cited = sum(1 for pid in cited_ids if is_stale(pid))
        stale_retrieved = sum(1 for pid in retrieved_ids if is_stale(pid))

        stale_rate = stale_cited / total_cited if total_cited > 0 else 0.0

        results.append({
            "question_id": gen["question_id"],
            "config": config_name,
            "method": gen.get("method", ""),
            "total_cited": total_cited,
            "stale_cited": stale_cited,
            "stale_citation_rate": round(stale_rate, 4),
            "stale_retrieved": stale_retrieved,
            "states_40_hours": "40 hours" in gen.get("answer", ""),
        })

    # Corpus freshness (descriptive, computed once)
    now = datetime.now()
    twelve_months_ago = now - timedelta(days=365)
    fresh_count = 0
    dated_count = 0
    for pid, p in collection.items():
        last_updated = p.get("last_updated", "")
        if last_updated:
            try:
                dt = datetime.fromisoformat(last_updated)
                dated_count += 1
                if dt >= twelve_months_ago:
                    fresh_count += 1
            except ValueError:
                pass

    if dated_count > 0:
        freshness = fresh_count / dated_count
        print(f"  Corpus freshness: {freshness:.1%} ({fresh_count}/{dated_count} passages within 12 months)")
    else:
        print("  Corpus freshness: N/A (no last_updated dates populated)")

    return results


def evaluate_all_configs():
    """Evaluate currency for all generation configs."""
    gen_dir = REPO_ROOT / "results" / "generations"
    all_results = []

    for gen_file in sorted(gen_dir.glob("*.jsonl")):
        config_name = gen_file.stem
        print(f"Evaluating currency: {config_name}")
        results = evaluate_currency(config_name)
        all_results.extend(results)

    if all_results:
        out_path = REPO_ROOT / "results" / "currency_metrics.csv"
        fields = list(all_results[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nWritten: {out_path}")
    else:
        print("\nNo currency results (need currency_sensitive=true questions + is_superseded passages)")

    return all_results


if __name__ == "__main__":
    evaluate_all_configs()
