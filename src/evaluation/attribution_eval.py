"""Attribution evaluation.

For each answer with citations, against qrels:
  - Citation precision: of cited passages, fraction judged relevant
  - Citation recall: of relevant passages in prompt, fraction cited
  - Citation validity: fraction of cited IDs that exist in the collection
  - Unattributed answer rate: answers making claims with no citation

Output: results/attribution_metrics.csv
"""

import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_collection_ids() -> set[str]:
    """Get all valid passage IDs from collection."""
    col_path = REPO_ROOT / "data" / "collection.jsonl"
    ids = set()
    with open(col_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                p = json.loads(line.strip())
                ids.add(p["id"])
    return ids


def load_qrels() -> dict[str, dict[str, int]]:
    """Load qrels → {question_id: {passage_id: relevance}}."""
    from src.evaluation.retrieval_eval import load_qrels as _load
    return _load()


def extract_citations(answer: str) -> list[str]:
    """Extract cited passage IDs from answer text.
    Matches patterns like [S01_001], [P001], [Source S03], etc.
    """
    # Match [S01_001] style
    cited = re.findall(r"\[S\d+_\d+\]", answer)
    cited = [c.strip("[]") for c in cited]

    # Also match [Source S03: ...] → extract S03
    source_refs = re.findall(r"\[Source (S\d+)", answer)

    return list(set(cited + source_refs))


def evaluate_attribution(config_name: str) -> list[dict]:
    """Evaluate attribution for a single configuration."""
    gen_path = REPO_ROOT / "results" / "generations" / f"{config_name}.jsonl"
    generations = []
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                generations.append(json.loads(line.strip()))

    collection_ids = load_collection_ids()
    qrels = load_qrels()
    results = []

    for gen in generations:
        qid = gen["question_id"]
        answer = gen["answer"]
        retrieved_ids = gen.get("retrieved_ids", [])

        if gen.get("refused", False):
            results.append({
                "question_id": qid,
                "question_type": gen["question_type"],
                "config": config_name,
                "num_citations": 0,
                "citation_precision": None,
                "citation_recall": None,
                "citation_validity": 1.0,
                "unattributed": False,
                "refused": True,
            })
            continue

        cited_ids = extract_citations(answer)

        # Citation validity: do cited IDs exist?
        if cited_ids:
            valid = sum(1 for c in cited_ids if c in collection_ids)
            citation_validity = valid / len(cited_ids)
        else:
            citation_validity = None

        # Citation precision: of cited passages, how many are relevant?
        qrel = qrels.get(qid, {})
        relevant_ids = {pid for pid, rel in qrel.items() if rel > 0}

        if cited_ids and relevant_ids:
            precision_hits = sum(1 for c in cited_ids if c in relevant_ids)
            citation_precision = precision_hits / len(cited_ids)
        elif cited_ids:
            citation_precision = 0.0
        else:
            citation_precision = None

        # Citation recall: of relevant passages in the prompt, how many were cited?
        relevant_in_prompt = relevant_ids & set(retrieved_ids)
        if relevant_in_prompt:
            recall_hits = sum(1 for r in relevant_in_prompt if r in cited_ids)
            citation_recall = recall_hits / len(relevant_in_prompt)
        else:
            citation_recall = None

        # Unattributed: answer has content but no citations
        has_content = len(answer.strip()) > 50
        unattributed = has_content and len(cited_ids) == 0

        results.append({
            "question_id": qid,
            "question_type": gen["question_type"],
            "config": config_name,
            "num_citations": len(cited_ids),
            "citation_precision": round(citation_precision, 4) if citation_precision is not None else None,
            "citation_recall": round(citation_recall, 4) if citation_recall is not None else None,
            "citation_validity": round(citation_validity, 4) if citation_validity is not None else None,
            "unattributed": unattributed,
            "refused": False,
        })

    return results


def evaluate_all_configs():
    """Evaluate attribution for all generation configs."""
    gen_dir = REPO_ROOT / "results" / "generations"
    all_results = []

    for gen_file in sorted(gen_dir.glob("*.jsonl")):
        config_name = gen_file.stem
        print(f"Evaluating attribution: {config_name}")
        results = evaluate_attribution(config_name)
        all_results.extend(results)

        valid_prec = [r["citation_precision"] for r in results if r["citation_precision"] is not None]
        unattr = sum(1 for r in results if r["unattributed"])
        print(f"  Mean citation precision: {sum(valid_prec)/len(valid_prec):.1%}" if valid_prec else "  No citations found")
        print(f"  Unattributed answers: {unattr}/{len(results)}")

    if all_results:
        out_path = REPO_ROOT / "results" / "attribution_metrics.csv"
        fields = list(all_results[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_results)
        print(f"\nWritten: {out_path}")

    return all_results


if __name__ == "__main__":
    evaluate_all_configs()
