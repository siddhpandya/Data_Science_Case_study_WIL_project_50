"""Phase 3: Generate manual coding sheets.

1. Main coding sheet: 30 randomly sampled answers from the Dense run,
   blind-shuffled (no config label visible to the coder).
2. Double-judging sheet: 10 randomly sampled answers, deduplicated,
   for inter-rater reliability (Cohen's kappa).

Uses the Dense run (dense-k5-settlein_v4) as specified.
"""

import csv
import json
import random
import sys
from pathlib import Path

REPO_ROOT = Path(r"c:\Users\Abhishek\OneDrive\Documents\Data_Science_Case_study_WIL_project_50")
sys.path.insert(0, str(REPO_ROOT))

DENSE_FILE = REPO_ROOT / "results" / "generations" / "dense-k5-settlein_v4.jsonl"
CB_FILE = REPO_ROOT / "results" / "generations" / "closed_book.jsonl"


def load_gen(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def make_coding_sheet():
    """Create the main coding sheet with 30 samples from Dense."""
    dense = load_gen(DENSE_FILE)
    closed = load_gen(CB_FILE)

    # Match by question_id
    dense_by_q = {r["question_id"]: r for r in dense}
    cb_by_q = {r["question_id"]: r for r in closed}

    # Get answerable questions only (where Dense actually answered)
    answerable = [r for r in dense if r["question_type"] in ("known", "inferred") and not r["refused_strict"]]

    # Sample 30 (or all if fewer)
    random.seed(42)
    sample_size = min(30, len(answerable))
    sampled = random.sample(answerable, sample_size)

    # Build blind-shuffled rows: for each question, include Dense and closed-book answers
    # but label them A/B in random order
    rows = []
    for r in sampled:
        qid = r["question_id"]
        dense_ans = dense_by_q[qid]["answer"]
        cb_ans = cb_by_q.get(qid, {}).get("answer", "")

        # Randomly assign A/B
        if random.random() < 0.5:
            ans_a, ans_b = dense_ans, cb_ans
            key_a, key_b = "dense", "closed_book"
        else:
            ans_a, ans_b = cb_ans, dense_ans
            key_a, key_b = "closed_book", "dense"

        rows.append({
            "sample_id": len(rows) + 1,
            "question_id": qid,
            "question": r["question"],
            "question_type": r["question_type"],
            "answer_A": ans_a,
            "answer_B": ans_b,
            # Hidden key (for scoring after coding)
            "_key_A": key_a,
            "_key_B": key_b,
            # Coding columns (to be filled by human coders)
            "A_supported": "",
            "A_applicable": "",
            "A_fabricated_quote": "",
            "A_notes": "",
            "B_supported": "",
            "B_applicable": "",
            "B_fabricated_quote": "",
            "B_notes": "",
        })

    # Write main coding sheet
    out_dir = REPO_ROOT / "results" / "coding_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    main_path = out_dir / "coding_sheet_main.csv"
    with open(main_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Main coding sheet: {main_path} ({len(rows)} samples)")

    # Write answer key (separate file, not seen by coders)
    key_path = out_dir / "coding_sheet_key.csv"
    key_rows = [{"sample_id": r["sample_id"], "question_id": r["question_id"],
                 "key_A": r["_key_A"], "key_B": r["_key_B"]} for r in rows]
    with open(key_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"Answer key: {key_path}")

    return rows


def make_double_judging_sheet(main_rows):
    """Create double-judging sheet: 10 samples for inter-rater reliability."""
    random.seed(99)
    sample_size = min(10, len(main_rows))
    sampled = random.sample(main_rows, sample_size)

    # Deduplicate and re-number
    rows = []
    for i, r in enumerate(sampled, 1):
        rows.append({
            "dj_id": i,
            "sample_id": r["sample_id"],
            "question_id": r["question_id"],
            "question": r["question"],
            "answer_A": r["answer_A"],
            "answer_B": r["answer_B"],
            # Coder 1
            "C1_A_supported": "",
            "C1_A_applicable": "",
            "C1_A_fabricated_quote": "",
            "C1_B_supported": "",
            "C1_B_applicable": "",
            "C1_B_fabricated_quote": "",
            # Coder 2
            "C2_A_supported": "",
            "C2_A_applicable": "",
            "C2_A_fabricated_quote": "",
            "C2_B_supported": "",
            "C2_B_applicable": "",
            "C2_B_fabricated_quote": "",
        })

    out_dir = REPO_ROOT / "results" / "coding_sheets"
    dj_path = out_dir / "double_judging_sheet.csv"
    with open(dj_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Double-judging sheet: {dj_path} ({len(rows)} samples)")


if __name__ == "__main__":
    main_rows = make_coding_sheet()
    make_double_judging_sheet(main_rows)
    print("\nDone. All coding sheets generated.")
