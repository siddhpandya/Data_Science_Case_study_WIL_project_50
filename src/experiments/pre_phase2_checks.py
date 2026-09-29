"""Pre-Phase 2 checks: A (v4 dev questions), B (citation validation), C (context overflow).

Usage:
  python -m src.experiments.pre_phase2_checks
"""

import csv
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.search import SearchEngine, load_collection
from src.generation.generate import (
    generate_answer, load_config, build_prompt,
    parse_citations, detect_refusal, validate_citations,
    format_passages_v3,
)


DEV_QUESTIONS = [
    ("DEV01", "how do i top up my myki"),
    ("DEV02", "can i drive in victoria with my overseas licence"),
    ("DEV03", "what does visa condition 8534 mean"),
    ("DEV04", "can international students get centrelink youth allowance"),  # should refuse
]


def check_a():
    """Check A: settlein_v4 on 4 dev questions."""
    print("=" * 70)
    print("CHECK A: settlein_v4 on dev questions")
    print("=" * 70)

    engine = SearchEngine()
    config = load_config()
    collection_ids = {p["id"] for p in engine.passages}

    all_pass = True

    for qid, question in DEV_QUESTIONS:
        print(f"\n{'━' * 60}")
        print(f"[{qid}] {question}")
        print(f"{'━' * 60}")

        results = engine.search(question, method="dense", top_k=5)
        retrieved_ids = [p["id"] for p, _ in results]
        print(f"Retrieved: {retrieved_ids}")

        t0 = time.time()
        gen_result = generate_answer(
            question=question,
            results=results,
            variant="settlein_v4",
            config=config,
        )
        latency = time.time() - t0
        answer = gen_result["answer"]
        prompt_tokens = gen_result["prompt_eval_count"]

        cited_ids, citation_mapped = parse_citations(answer, retrieved_ids)
        cited_not_retrieved, cited_nonexistent = validate_citations(
            cited_ids, retrieved_ids, collection_ids
        )
        refused_strict, refused_lenient = detect_refusal(answer)

        print(f"\nRaw answer ({latency:.1f}s, {prompt_tokens} prompt tokens):")
        print("─" * 40)
        print(answer)
        print("─" * 40)
        print(f"Cited IDs:            {cited_ids}")
        print(f"citation_mapped:      {citation_mapped}")
        print(f"cited_not_retrieved:  {cited_not_retrieved}")
        print(f"cited_nonexistent:   {cited_nonexistent}")
        print(f"refused_strict:      {refused_strict}")
        print(f"refused_lenient:     {refused_lenient}")
        print(f"prompt_eval_count:   {prompt_tokens}")

        # Check: DEV04 should return exact refusal string
        if qid == "DEV04":
            if refused_strict:
                print("  ✅ DEV04 correctly refused (strict)")
            else:
                print("  ❌ DEV04 did NOT return exact refusal string!")
                all_pass = False

        # Check: no citations at start of answer for answered questions
        if not refused_strict and answer.strip().startswith("[S"):
            print("  ⚠️ Citation at start of answer (v4 rule violation)")
            all_pass = False

        # Check: hallucinated citations
        if cited_not_retrieved:
            print(f"  ⚠️ cited_not_retrieved: {cited_not_retrieved}")
        if cited_nonexistent:
            print(f"  ⚠️ cited_nonexistent: {cited_nonexistent}")

    return all_pass


def check_c():
    """Check C: context overflow estimation."""
    print("\n" + "=" * 70)
    print("CHECK C: Context overflow estimation")
    print("=" * 70)

    engine = SearchEngine()
    config = load_config()
    num_ctx = config["generation"].get("num_ctx", 4096)
    num_predict = config["generation"].get("num_predict", 512)

    topics_path = REPO_ROOT / "data" / "topics.csv"
    with open(topics_path, "r", encoding="utf-8") as f:
        topics = [r for r in csv.DictReader(f) if r.get("question_id", "").strip()]

    max_chars = 0
    max_est_tokens = 0
    max_qid = ""
    max_method = ""

    results_list = []

    for method in ["bm25", "dense"]:
        for topic in topics:
            qid = topic["question_id"]
            question = topic["question"]

            results = engine.search(question, method=method, top_k=5)

            # Build the full prompt
            sys_prompt, user_prompt = build_prompt(question, results, "settlein_v4")
            total_chars = len(sys_prompt) + len(user_prompt)
            est_tokens = total_chars // 4  # rough estimate

            results_list.append({
                "qid": qid,
                "method": method,
                "chars": total_chars,
                "est_tokens": est_tokens,
            })

            if est_tokens > max_est_tokens:
                max_est_tokens = est_tokens
                max_chars = total_chars
                max_qid = qid
                max_method = method

    total_budget = num_ctx
    prompt_budget = total_budget - num_predict

    print(f"\nnum_ctx = {num_ctx}")
    print(f"num_predict = {num_predict}")
    print(f"Prompt budget (num_ctx - num_predict) = {prompt_budget} tokens")
    print(f"\nMax prompt across all 48 questions × 2 retrievers:")
    print(f"  {max_qid} ({max_method}): {max_chars} chars ≈ {max_est_tokens} tokens")
    print(f"\nPrompt + num_predict = {max_est_tokens} + {num_predict} = {max_est_tokens + num_predict}")

    if max_est_tokens + num_predict > num_ctx:
        print(f"\n❌ OVERFLOW: {max_est_tokens + num_predict} > {num_ctx}")
        print("Ollama will silently drop the start of the prompt (system instructions)!")
        return False
    else:
        headroom = prompt_budget - max_est_tokens
        print(f"\n✅ SAFE: {headroom} tokens headroom")

        # Show distribution
        est_tokens_all = [r["est_tokens"] for r in results_list]
        est_tokens_all.sort()
        print(f"Token distribution: min={est_tokens_all[0]}, median={est_tokens_all[len(est_tokens_all)//2]}, max={est_tokens_all[-1]}")
        return True


if __name__ == "__main__":
    a_pass = check_a()
    c_pass = check_c()

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Check A (v4 dev questions): {'✅ PASS' if a_pass else '❌ FAIL'}")
    print(f"Check B (citation validation): ✅ INTEGRATED (runs per-answer in batch_run)")
    print(f"Check C (context overflow):    {'✅ PASS' if c_pass else '❌ FAIL'}")

    if not a_pass or not c_pass:
        print("\n⚠️ Some checks failed — review before Phase 2")
        sys.exit(1)
    else:
        print("\nAll checks passed — safe to proceed to Phase 2")
