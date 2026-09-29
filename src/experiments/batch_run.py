"""Batch pipeline: run all topics through retrieval and generation.

Produces:
  - TREC run files in runs/{config_name}.txt
  - Generation outputs in results/generations/{config_name}.jsonl

Resumable: skips questions already in the output JSONL.
"""

import csv
import json
import time
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.search import SearchEngine, load_collection
from src.generation.generate import (
    generate_answer, load_config,
    parse_citations, detect_refusal, validate_citations,
)


def load_topics() -> list[dict]:
    """Load topics.csv."""
    topics_path = REPO_ROOT / "data" / "topics.csv"
    with open(topics_path, "r", encoding="utf-8") as f:
        return [r for r in csv.DictReader(f) if r.get("question_id", "").strip()]


def load_existing_generations(path: Path) -> set[str]:
    """Load question IDs already generated (for resumability)."""
    if not path.exists():
        return set()
    done = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line.strip())
                done.add(rec["question_id"])
    return done


def run_retrieval_batch(
    engine: SearchEngine,
    topics: list[dict],
    method: str,
    top_k: int,
    run_tag: str,
) -> dict[str, list[tuple[dict, float]]]:
    """Run retrieval for all topics, write TREC run file, return results."""
    runs_dir = REPO_ROOT / "runs"
    runs_dir.mkdir(exist_ok=True)
    run_file = runs_dir / f"{run_tag}.txt"

    all_results = {}

    with open(run_file, "w", encoding="utf-8") as f:
        for topic in topics:
            qid = topic["question_id"]
            question = topic["question"]

            results = engine.search(question, method=method, top_k=top_k)
            all_results[qid] = results

            for rank, (passage, score) in enumerate(results, 1):
                pid = passage["id"]
                f.write(f"{qid} Q0 {pid} {rank} {score:.6f} {run_tag}\n")

    print(f"  TREC run file: {run_file} ({len(topics)} queries)")
    return all_results


def run_generation_batch(
    topics: list[dict],
    retrieval_results: dict[str, list[tuple[dict, float]]],
    variant: str,
    config_name: str,
    config: dict | None = None,
    collection_ids: set[str] | None = None,
) -> list[dict]:
    """Run generation for all topics, write generations JSONL.

    Resumable: appends to existing file, skips already-done questions.
    """
    if config is None:
        config = load_config()

    gen_dir = REPO_ROOT / "results" / "generations"
    gen_dir.mkdir(parents=True, exist_ok=True)
    out_path = gen_dir / f"{config_name}.jsonl"

    # Load already-done question IDs for resumability
    done_qids = load_existing_generations(out_path)
    if done_qids:
        print(f"  Resuming: {len(done_qids)} already done, skipping them")

    generations = []

    # Open in append mode for resumability
    with open(out_path, "a", encoding="utf-8") as f:
        for topic in topics:
            qid = topic["question_id"]
            if qid in done_qids:
                continue

            question = topic["question"]
            q_type = topic.get("question_type", "known")

            results = retrieval_results.get(qid)
            retrieved_ids = [r[0]["id"] for r in results] if results else []

            t0 = time.time()
            gen_result = generate_answer(
                question=question,
                results=results,
                variant=variant,
                config=config,
            )
            latency_ms = int((time.time() - t0) * 1000)

            answer = gen_result["answer"]
            prompt_eval_count = gen_result["prompt_eval_count"]
            eval_count = gen_result["eval_count"]

            # Parse citations
            cited_ids, citation_mapped = parse_citations(answer, retrieved_ids)

            # Validate citations
            cited_not_retrieved, cited_nonexistent = validate_citations(
                cited_ids, retrieved_ids, collection_ids
            )

            # Detect refusal
            refused_strict, refused_lenient = detect_refusal(answer)

            gen_record = {
                "question_id": qid,
                "question": question,
                "question_type": q_type,
                "answer": answer,
                "retrieved_ids": retrieved_ids,
                "retrieved_scores": [round(r[1], 6) for r in (results or [])],
                "cited_ids": cited_ids,
                "citation_mapped": citation_mapped,
                "cited_not_retrieved": cited_not_retrieved,
                "cited_nonexistent": cited_nonexistent,
                "refused_strict": refused_strict,
                "refused_lenient": refused_lenient,
                "refused": refused_strict,  # backward compat
                "refusal_mechanism": "prompt" if refused_strict else "none",
                "latency_ms": latency_ms,
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
                "config": config_name,
            }
            generations.append(gen_record)

            # Write immediately (crash-safe)
            f.write(json.dumps(gen_record, ensure_ascii=False) + "\n")
            f.flush()

            status = "REFUSED" if refused_strict else f"{len(answer)} chars"
            warns = ""
            if cited_not_retrieved:
                warns += f" ⚠ not_retrieved={cited_not_retrieved}"
            if cited_nonexistent:
                warns += f" ⚠ nonexistent={cited_nonexistent}"
            print(f"    [{qid}] {status} ({latency_ms}ms) cited={cited_ids} prompt_tokens={prompt_eval_count}{warns}")

    total = len(done_qids) + len(generations)
    print(f"  Generations: {out_path} ({total} answers total, {len(generations)} new)")
    return generations


def run_single_config(
    method: str,
    top_k: int,
    variant: str,
    config_name: str | None = None,
):
    """Run a single retrieval+generation configuration."""
    if config_name is None:
        config_name = f"{method}-k{top_k}-{variant}"

    print(f"\n{'='*60}")
    print(f"CONFIG: {config_name}")
    print(f"  method={method}, top_k={top_k}, variant={variant}")
    print(f"{'='*60}")

    topics = load_topics()
    if not topics:
        print("ERROR: No topics in topics.csv", file=sys.stderr)
        return

    print(f"  Loaded {len(topics)} topics")

    # Load engine
    engine = SearchEngine()

    # Build collection ID set for citation validation
    collection_ids = {p["id"] for p in engine.passages}

    # Retrieval
    if method == "closed_book":
        retrieval_results = {t["question_id"]: None for t in topics}
        print("  Skipping retrieval (closed-book)")
    else:
        retrieval_results = run_retrieval_batch(
            engine, topics, method, top_k, config_name
        )

    # Generation
    config = load_config()
    generations = run_generation_batch(
        topics, retrieval_results, variant, config_name, config,
        collection_ids=collection_ids,
    )

    return generations


if __name__ == "__main__":
    import argparse
    from src.generation.prompts import PROMPTS

    parser = argparse.ArgumentParser(description="Run batch retrieval + generation")
    parser.add_argument("--method", default="dense", choices=["bm25", "dense", "closed_book"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--variant", default="settlein_v4", choices=list(PROMPTS.keys()))
    parser.add_argument("--name", default=None, help="Config name override")
    args = parser.parse_args()

    run_single_config(args.method, args.top_k, args.variant, args.name)

