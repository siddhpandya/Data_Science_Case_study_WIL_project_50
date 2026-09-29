"""Phase 3: Generate corrected coding sheets per spec.

Coding sheet (60 rows):
  - 30 questions: 15 Known, all 6 Inferred, 9 Out-of-KB
  - Each question appears twice: Dense answer + bare closed-book answer
  - Shuffled, no indication of which run
  - Columns: question, answer, retrieved_passages, gold_answer,
             correct, supported, applicable, citation_valid,
             fabricated_quote (pre-filled), notes

Double-judging sheet (relevance):
  - 10 answerable questions (at least 1 from each topic area)
  - Combined BM25 top-10 + Dense top-10 passages, deduplicated, shuffled
  - Full passage text, no scores, no labels
  - Empty label column (2, 1, or blank)
"""

import csv
import json
import random
import re
import sys
from pathlib import Path
from collections import defaultdict

REPO = Path(r"c:\Users\Abhishek\OneDrive\Documents\Data_Science_Case_study_WIL_project_50")
sys.path.insert(0, str(REPO))


def load_gen(name):
    return [json.loads(l) for l in open(REPO / "results" / "generations" / f"{name}.jsonl", encoding="utf-8")]


def load_topics():
    with open(REPO / "data" / "topics.csv", encoding="utf-8") as f:
        return {r["question_id"].strip(): r for r in csv.DictReader(f)}


def load_gold():
    with open(REPO / "data" / "gold_answers.csv", encoding="utf-8") as f:
        return {r["question_id"].strip(): r for r in csv.DictReader(f)}


def load_collection():
    passages = {}
    with open(REPO / "data" / "collection.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            passages[p["id"]] = p["contents"]
    return passages


def has_fabricated_quote(answer):
    """Check if answer contains quoted text (potential fabrication)."""
    return bool(re.search(r'"[^"]{20,}"', answer))


def make_coding_sheet():
    """Create 60-row coding sheet: 30 Dense + 30 bare closed-book."""
    dense = load_gen("dense-k5-settlein_v4")
    cb = load_gen("closed_book")
    topics = load_topics()
    gold = load_gold()

    dense_by_q = {r["question_id"]: r for r in dense}
    cb_by_q = {r["question_id"]: r for r in cb}

    # Select 30 questions: 15 Known, all 6 Inferred, 9 Out-of-KB
    known = [r for r in dense if r["question_type"] == "known"]
    inferred = [r for r in dense if r["question_type"] == "inferred"]
    out_of_kb = [r for r in dense if r["question_type"] == "out_of_kb"]

    random.seed(42)
    selected_known = random.sample(known, 15)
    selected_inferred = inferred  # all 6
    selected_ookb = random.sample(out_of_kb, 9)

    selected = selected_known + selected_inferred + selected_ookb
    selected_qids = [r["question_id"] for r in selected]

    # Build 60 rows: each question gets Dense row and CB row
    rows = []
    for qid in selected_qids:
        d = dense_by_q[qid]
        c = cb_by_q[qid]
        g = gold.get(qid, {})

        # Dense row
        rows.append({
            "question": d["question"],
            "answer": d["answer"],
            "retrieved_passages": "; ".join(d.get("retrieved_ids", [])),
            "gold_answer": g.get("gold_answer", ""),
            "correct": "",
            "supported": "",
            "applicable": "",
            "citation_valid": "",
            "fabricated_quote": "YES" if has_fabricated_quote(d["answer"]) else "",
            "notes": "",
        })

        # Closed-book row (no retrieved passages)
        rows.append({
            "question": c["question"],
            "answer": c["answer"],
            "retrieved_passages": "",
            "gold_answer": g.get("gold_answer", ""),
            "correct": "",
            "supported": "",
            "applicable": "",
            "citation_valid": "",
            "fabricated_quote": "YES" if has_fabricated_quote(c["answer"]) else "",
            "notes": "",
        })

    # Shuffle all 60 rows
    random.shuffle(rows)

    # Write two identical copies
    out_dir = REPO / "results" / "coding_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())

    for coder in ["coder1", "coder2"]:
        path = out_dir / f"coding_sheet_{coder}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  {path} ({len(rows)} rows)")

    # Write answer key (separate, hidden from coders)
    # Reset RNG to match the shuffle
    random.seed(42)
    key_rows = []
    for qid in selected_qids:
        key_rows.append({"question_id": qid, "row_type": "dense"})
        key_rows.append({"question_id": qid, "row_type": "closed_book"})
    random.shuffle(key_rows)  # same shuffle as above

    # Wait - the shuffle above is different because we set seed earlier.
    # Actually we need to track which row is which. Let me use a different approach.
    # Build rows with hidden labels, shuffle, then extract labels.
    random.seed(42)
    labeled_rows = []
    for qid in selected_qids:
        d = dense_by_q[qid]
        c = cb_by_q[qid]
        labeled_rows.append({"question_id": qid, "source": "dense", "question": d["question"]})
        labeled_rows.append({"question_id": qid, "source": "closed_book", "question": c["question"]})
    random.shuffle(labeled_rows)  # Different state than the rows shuffle above!

    # This won't match. Let me rebuild properly.
    pass  # We'll use a single-pass approach instead.


def make_coding_sheet_v2():
    """Create 60-row coding sheet with proper key tracking."""
    dense = load_gen("dense-k5-settlein_v4")
    cb = load_gen("closed_book")
    topics = load_topics()
    gold = load_gold()

    dense_by_q = {r["question_id"]: r for r in dense}
    cb_by_q = {r["question_id"]: r for r in cb}

    # Select 30 questions: 15 Known, all 6 Inferred, 9 Out-of-KB
    known_qids = [r["question_id"] for r in dense if r["question_type"] == "known"]
    inferred_qids = [r["question_id"] for r in dense if r["question_type"] == "inferred"]
    ookb_qids = [r["question_id"] for r in dense if r["question_type"] == "out_of_kb"]

    random.seed(42)
    selected_known = random.sample(known_qids, 15)
    selected_inferred = inferred_qids  # all 6
    selected_ookb = random.sample(ookb_qids, 9)

    selected_qids = selected_known + selected_inferred + selected_ookb
    print(f"  Selected: {len(selected_known)} known, {len(selected_inferred)} inferred, {len(selected_ookb)} out_of_kb")

    # Build 60 rows with hidden source tag for the key
    all_rows = []
    for qid in selected_qids:
        d = dense_by_q[qid]
        c = cb_by_q[qid]
        g = gold.get(qid, {})

        for source, rec in [("dense", d), ("closed_book", c)]:
            retrieved = "; ".join(rec.get("retrieved_ids", [])) if source == "dense" else ""
            all_rows.append({
                "_source": source,
                "_question_id": qid,
                "question": rec["question"],
                "answer": rec["answer"],
                "retrieved_passages": retrieved,
                "gold_answer": g.get("gold_answer", ""),
                "correct": "",
                "supported": "",
                "applicable": "",
                "citation_valid": "",
                "fabricated_quote": "YES" if has_fabricated_quote(rec["answer"]) else "",
                "notes": "",
            })

    # Shuffle
    random.seed(99)
    random.shuffle(all_rows)

    # Extract key before stripping hidden fields
    key_rows = [{"row_number": i+1, "question_id": r["_question_id"], "source": r["_source"]}
                for i, r in enumerate(all_rows)]

    # Strip hidden fields for coder sheets
    public_fields = [k for k in all_rows[0].keys() if not k.startswith("_")]
    public_rows = [{k: r[k] for k in public_fields} for r in all_rows]

    out_dir = REPO / "results" / "coding_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    for coder in ["coder1", "coder2"]:
        path = out_dir / f"coding_sheet_{coder}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=public_fields)
            writer.writeheader()
            writer.writerows(public_rows)
        print(f"  {path} ({len(public_rows)} rows)")

    # Write key
    key_path = out_dir / "coding_sheet_key.csv"
    with open(key_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"  Key: {key_path}")

    return selected_qids


def make_relevance_judging_sheet():
    """Create double-judging sheet for passage relevance."""
    topics = load_topics()
    collection = load_collection()

    from src.retrieval.search import SearchEngine
    engine = SearchEngine()

    # For each selected question, retrieve BM25 top-10 and Dense top-10
    def get_top10(question, method):
        results = engine.search(question, method=method, top_k=10)
        return [p["id"] for p, _ in results]

    # Pick 10 answerable questions, at least 1 from each of 6 topic areas
    # Topic areas: work rights (T01-T05), health (T06-T11), housing (T12-T16),
    #              transport (T17-T19), money (T20-T31), admin (T32-T42)
    areas = {
        "work_rights": ["T01", "T02", "T03", "T04", "T05"],
        "health": ["T06", "T07", "T08", "T09", "T10", "T11"],
        "housing": ["T12", "T13", "T14", "T15", "T16"],
        "transport": ["T17", "T18", "T19"],
        "money": ["T20", "T21", "T22", "T23", "T24", "T25", "T26"],
        "admin": ["T27", "T28", "T29", "T30", "T31", "T32", "T33", "T34", "T35", "T36", "T37", "T38", "T39", "T40", "T41", "T42"],
    }

    answerable = {qid: t for qid, t in topics.items() if t["question_type"] in ("known", "inferred")}

    random.seed(77)
    selected = []
    used_topics = set()

    # First, pick one from each area
    for area_name, tid_prefixes in areas.items():
        candidates = [qid for qid in answerable if any(qid.startswith(tp) for tp in tid_prefixes)]
        if candidates:
            pick = random.choice(candidates)
            selected.append(pick)
            used_topics.add(pick[:3])  # T01, T02 etc.

    # Fill remaining slots
    remaining = [qid for qid in answerable if qid not in selected]
    random.shuffle(remaining)
    for qid in remaining:
        if len(selected) >= 10:
            break
        selected.append(qid)

    print(f"\n  Relevance judging: {len(selected)} questions from areas: {sorted(set(q[:3] for q in selected))}")

    # Build rows: for each question, combine BM25 top-10 + Dense top-10, dedup, shuffle
    all_rows = []
    for qid in sorted(selected):
        q_text = topics[qid]["question"]

        bm25_pids = get_top10(q_text, "bm25")
        dense_pids = get_top10(q_text, "dense")

        # Deduplicate, preserving insertion order
        combined = []
        seen = set()
        for pid in bm25_pids + dense_pids:
            if pid not in seen:
                combined.append(pid)
                seen.add(pid)

        # Shuffle
        random.shuffle(combined)

        for pid in combined:
            all_rows.append({
                "question_id": qid,
                "question": q_text,
                "passage_id": pid,
                "passage_text": collection.get(pid, "[TEXT NOT FOUND]"),
                "label": "",  # 2, 1, or blank
            })

    out_dir = REPO / "results" / "coding_sheets"
    path = out_dir / "relevance_judging_sheet.csv"
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"  {path} ({len(all_rows)} passage judgments for {len(selected)} questions)")


if __name__ == "__main__":
    print("=" * 70)
    print("CODING SHEETS")
    print("=" * 70)
    make_coding_sheet_v2()

    print("\n" + "=" * 70)
    print("RELEVANCE JUDGING SHEET")
    print("=" * 70)
    make_relevance_judging_sheet()

    print("\nDone.")
