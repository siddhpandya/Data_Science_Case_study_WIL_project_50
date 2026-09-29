"""Phase 1 verification: generation, citations, refusal, speed.

Items tested:
  2. Strict/lenient refusal detector (unit test).
  5. Median latency over 5 answers.
  6. Full output for T02Q01, T05Q01, T06Q01, T07Q01.

Usage:
  python -m src.experiments.phase1_test
"""

import csv
import json
import re
import sys
import time
from pathlib import Path
from statistics import median

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.retrieval.search import SearchEngine, load_config
from src.generation.generate import (
    generate_answer, parse_citations, detect_refusal, REFUSAL_STRING,
)


def test_refusal_detector():
    """Item 2: Verify strict/lenient detector on specific inputs."""
    print("=" * 70)
    print("ITEM 2: Refusal detector unit tests")
    print("=" * 70)

    # Test case from the spec
    test_answer = (
        "I don't have specific information about a detailed roadmap for the "
        "first week in university. However, I recommend checking with your "
        "education provider."
    )
    strict, lenient = detect_refusal(test_answer)
    print(f"\nTest: \"{test_answer[:80]}...\"")
    print(f"  strict = {strict} (expected: False)")
    print(f"  lenient = {lenient} (expected: True)")
    assert strict is False, f"FAIL: strict should be False, got {strict}"
    assert lenient is True, f"FAIL: lenient should be True, got {lenient}"
    print("  ✅ PASS")

    # Exact refusal string
    strict2, lenient2 = detect_refusal(REFUSAL_STRING)
    print(f"\nTest: exact refusal string")
    print(f"  strict = {strict2} (expected: True)")
    print(f"  lenient = {lenient2} (expected: True)")
    assert strict2 is True, f"FAIL: strict should be True"
    assert lenient2 is True, f"FAIL: lenient should be True"
    print("  ✅ PASS")

    # Normal answer (no refusal)
    normal = "You can work up to 48 hours a fortnight while your course is in session [S03_002]."
    strict3, lenient3 = detect_refusal(normal)
    print(f"\nTest: normal answer")
    print(f"  strict = {strict3} (expected: False)")
    print(f"  lenient = {lenient3} (expected: False)")
    assert strict3 is False
    assert lenient3 is False
    print("  ✅ PASS")

    print("\nAll refusal detector tests passed.\n")


def test_citation_parser():
    """Verify citation parsing."""
    print("=" * 70)
    print("ITEM 1: Citation parser tests")
    print("=" * 70)

    # Direct passage IDs
    answer1 = "You can work 48 hours [S03_002] during session [S17_001]."
    ids1, mapped1 = parse_citations(answer1)
    print(f"\nTest: direct passage IDs")
    print(f"  cited = {ids1} (expected: ['S03_002', 'S17_001'])")
    print(f"  mapped = {mapped1} (expected: False)")
    assert ids1 == ["S03_002", "S17_001"], f"FAIL: {ids1}"
    assert mapped1 is False
    print("  ✅ PASS")

    # Fallback: Passage N
    answer2 = "According to Passage 1 and Passage 3, you should apply."
    ids2, mapped2 = parse_citations(answer2, ["S03_002", "S17_001", "S05_003"])
    print(f"\nTest: Passage N fallback")
    print(f"  cited = {ids2} (expected: ['S03_002', 'S05_003'])")
    print(f"  mapped = {mapped2} (expected: True)")
    assert ids2 == ["S03_002", "S05_003"], f"FAIL: {ids2}"
    assert mapped2 is True
    print("  ✅ PASS")

    # No citations
    answer3 = "I have no idea."
    ids3, mapped3 = parse_citations(answer3)
    print(f"\nTest: no citations")
    print(f"  cited = {ids3} (expected: [])")
    print(f"  mapped = {mapped3} (expected: False)")
    assert ids3 == []
    assert mapped3 is False
    print("  ✅ PASS")

    print("\nAll citation parser tests passed.\n")


def run_generation_test():
    """Items 5 & 6: Latency measurement and full output for 4 questions."""
    print("=" * 70)
    print("ITEM 5 & 6: Generation test")
    print("=" * 70)

    # Load topics
    target_qids = ["T02Q01", "T05Q01", "T06Q01", "T07Q01"]
    topics = {}
    with open(REPO_ROOT / "data" / "topics.csv", "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qid = r.get("question_id", "").strip()
            if qid in target_qids:
                topics[qid] = r

    engine = SearchEngine()
    config = load_config()

    print(f"\nModel: {config['generation']['model']}")
    print(f"num_ctx: {config['generation'].get('num_ctx')}")
    print(f"seed: {config['generation'].get('seed')}")
    print(f"prompt_variant: {config['generation'].get('prompt_variant')}")
    print(f"Passages: {len(engine.passages)}")

    # Item 5: time 5 answers for median latency
    # Use the first 5 questions from topics.csv
    speed_topics = []
    with open(REPO_ROOT / "data" / "topics.csv", "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("question_type") == "known" and len(speed_topics) < 5:
                speed_topics.append(r)

    latencies = []
    print(f"\n--- Item 5: Speed test (5 answers) ---")
    for i, topic in enumerate(speed_topics, 1):
        q = topic["question"]
        results = engine.search(q, method="dense", top_k=5)
        t0 = time.time()
        answer = generate_answer(question=q, results=results, variant="settlein_v3", config=config)
        lat = time.time() - t0
        latencies.append(lat)
        print(f"  [{i}/5] {topic['question_id']}: {lat:.1f}s ({len(answer)} chars)")

    med = median(latencies)
    print(f"\n  Median latency: {med:.1f}s")
    print(f"  All latencies: {[round(l, 1) for l in latencies]}")

    # Item 6: Full output for target questions
    print(f"\n--- Item 6: Full output for {', '.join(target_qids)} ---")

    for qid in target_qids:
        topic = topics[qid]
        q = topic["question"]
        q_type = topic["question_type"]

        print(f"\n{'━' * 70}")
        print(f"[{qid}] ({q_type})")
        print(f"Q: {q}")
        print(f"{'━' * 70}")

        # Retrieve
        results = engine.search(q, method="dense", top_k=5)
        retrieved_ids = [p["id"] for p, _ in results]
        print(f"Retrieved: {retrieved_ids}")

        # Generate
        t0 = time.time()
        answer = generate_answer(question=q, results=results, variant="settlein_v3", config=config)
        latency = time.time() - t0

        # Parse
        cited_ids, citation_mapped = parse_citations(answer, retrieved_ids)
        refused_strict, refused_lenient = detect_refusal(answer)

        print(f"\nRaw answer ({latency:.1f}s):")
        print("─" * 40)
        print(answer)
        print("─" * 40)
        print(f"Cited IDs:       {cited_ids}")
        print(f"citation_mapped: {citation_mapped}")
        print(f"refused_strict:  {refused_strict}")
        print(f"refused_lenient: {refused_lenient}")
        print(f"Latency:         {latency:.1f}s")


if __name__ == "__main__":
    test_refusal_detector()
    test_citation_parser()
    run_generation_test()
