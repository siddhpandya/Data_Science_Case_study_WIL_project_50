"""Currency generation with contextual headers (settlein_v4_ctx).

Runs T01Q01, T02Q01, T03Q01 with BM25 and Dense at k=5, include_superseded=true.
Saves to results/generations/currency_ctx.jsonl.
"""

import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(r"c:\Users\Abhishek\OneDrive\Documents\Data_Science_Case_study_WIL_project_50")
sys.path.insert(0, str(REPO))

from src.retrieval.search import load_collection, SearchEngine, filter_superseded
from src.generation.generate import generate_answer, load_config, detect_refusal, parse_citations


def load_topics():
    with open(REPO / "data" / "topics.csv", encoding="utf-8") as f:
        return {r["question_id"].strip(): r for r in csv.DictReader(f)}


def main():
    currency_qids = ["T01Q01", "T02Q01", "T03Q01"]
    topics = load_topics()
    config = load_config()

    all_passages = load_collection()
    # include_superseded=True for currency experiment
    engine = SearchEngine(all_passages, include_superseded=True)

    results = []
    for method in ["bm25", "dense"]:
        for qid in currency_qids:
            t = topics[qid]
            question = t["question"]

            retrieved = engine.search(question, method=method, top_k=5)
            retrieved_ids = [p["id"] for p, _ in retrieved]

            gen_result = generate_answer(
                question=question,
                results=retrieved,
                variant="settlein_v4_ctx",
                config=config,
            )
            answer = gen_result["answer"]
            refused_strict, refused_lenient = detect_refusal(answer)
            cited_ids, _ = parse_citations(answer, retrieved_ids)

            rec = {
                "question_id": qid,
                "question": question,
                "question_type": t.get("question_type", ""),
                "method": method,
                "variant": "settlein_v4_ctx",
                "retrieved_ids": retrieved_ids,
                "cited_ids": cited_ids,
                "refused_strict": refused_strict,
                "refused_lenient": refused_lenient,
                "answer": answer,
                "contains_40_hours": "40 hours" in answer.lower() or "40-hour" in answer.lower(),
            }
            results.append(rec)

            print(f"\n{'='*70}")
            print(f"[{method.upper()}] {qid}: {question}")
            print(f"Retrieved: {retrieved_ids}")
            print(f"Cited: {cited_ids}")
            print(f"Refused: strict={refused_strict}, lenient={refused_lenient}")
            print(f"Contains '40 hours': {rec['contains_40_hours']}")
            print(f"--- ANSWER ---")
            print(answer)

    # Save
    out_path = REPO / "results" / "generations" / "currency_ctx.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved: {out_path}")

    # Compare with originals
    print(f"\n{'='*70}")
    print("COMPARISON: 40-hour mentions")
    print(f"{'='*70}")
    for fname, label in [
        ("currency_with_superseded.jsonl", "Original Dense"),
        ("currency_bm25_with_superseded.jsonl", "Original BM25"),
    ]:
        orig_path = REPO / "results" / "generations" / fname
        if orig_path.exists():
            for line in open(orig_path, encoding="utf-8"):
                r = json.loads(line)
                has_40 = "40 hours" in r["answer"].lower() or "40-hour" in r["answer"].lower()
                print(f"  [{label}] {r['question_id']}: contains '40 hours' = {has_40}")

    print("\n  Contextual (v4_ctx):")
    for r in results:
        print(f"  [{r['method'].upper()} ctx] {r['question_id']}: contains '40 hours' = {r['contains_40_hours']}")


if __name__ == "__main__":
    main()
