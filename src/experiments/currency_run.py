"""Currency run: generate answers for T01Q01, T02Q01, T03Q01 with superseded passages included.

This produces a separate generation file to compare answers with/without S18 passages.
"""

import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.search import SearchEngine
from src.generation.generate import (
    generate_answer, load_config,
    parse_citations, detect_refusal, validate_citations,
)


CURRENCY_QIDS = {"T01Q01", "T02Q01", "T03Q01"}


def main():
    print("=" * 70)
    print("CURRENCY RUN: include_superseded=true")
    print("=" * 70)

    config = load_config()

    # Load ALL passages including superseded
    col_path = REPO_ROOT / "data" / "collection.jsonl"
    all_passages = []
    with open(col_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                all_passages.append(json.loads(line.strip()))

    print(f"All passages (including superseded): {len(all_passages)}")

    # Build a search engine WITHOUT the superseded filter
    # We'll do this by creating a custom engine
    from src.retrieval.search import SearchEngine
    engine = SearchEngine(include_superseded=True)
    print(f"Engine passages (with superseded): {len(engine.passages)}")

    collection_ids = {p["id"] for p in engine.passages}

    # Load topics
    topics_path = REPO_ROOT / "data" / "topics.csv"
    with open(topics_path, "r", encoding="utf-8") as f:
        topics = [r for r in csv.DictReader(f) if r.get("question_id", "").strip() in CURRENCY_QIDS]

    print(f"Currency questions: {len(topics)}")

    gen_dir = REPO_ROOT / "results" / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    out_path = gen_dir / "currency_with_superseded.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for topic in topics:
            qid = topic["question_id"]
            question = topic["question"]

            # Dense retrieval with superseded included
            results = engine.search(question, method="dense", top_k=5)
            retrieved_ids = [p["id"] for p, _ in results]
            has_superseded = any(pid.startswith("S18") for pid in retrieved_ids)

            print(f"\n[{qid}] {question}")
            print(f"  Retrieved: {retrieved_ids}")
            print(f"  Has S18 passage: {has_superseded}")

            t0 = time.time()
            gen_result = generate_answer(
                question=question,
                results=results,
                variant="settlein_v4",
                config=config,
            )
            latency_ms = int((time.time() - t0) * 1000)

            answer = gen_result["answer"]
            cited_ids, citation_mapped = parse_citations(answer, retrieved_ids)
            cited_not_retrieved, cited_nonexistent = validate_citations(
                cited_ids, retrieved_ids, collection_ids
            )
            refused_strict, refused_lenient = detect_refusal(answer)

            record = {
                "question_id": qid,
                "question": question,
                "answer": answer,
                "retrieved_ids": retrieved_ids,
                "has_superseded_passage": has_superseded,
                "cited_ids": cited_ids,
                "cited_superseded": [c for c in cited_ids if c.startswith("S18")],
                "cited_not_retrieved": cited_not_retrieved,
                "cited_nonexistent": cited_nonexistent,
                "refused_strict": refused_strict,
                "latency_ms": latency_ms,
                "prompt_eval_count": gen_result["prompt_eval_count"],
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

            print(f"  Answer: {answer[:100]}...")
            print(f"  Cited: {cited_ids}")
            print(f"  Cited superseded: {record['cited_superseded']}")
            print(f"  Latency: {latency_ms}ms")

    print(f"\nWritten: {out_path}")


if __name__ == "__main__":
    main()
