"""SettleIN evaluation modules.

Shared utilities for loading gold answers and other evaluation data.
"""

import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def load_gold_answers() -> dict[str, dict]:
    """Load gold_answers.csv → {question_id: {gold_answer, passage_ids}}.

    passage_ids is split on semicolons into a list.
    """
    gold_path = REPO_ROOT / "data" / "gold_answers.csv"
    golds = {}
    with open(gold_path, "r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qid = r.get("question_id", "").strip()
            if not qid:
                continue
            raw_ids = r.get("passage_ids", "").strip()
            # Split on semicolons, strip whitespace from each ID
            passage_ids = [pid.strip() for pid in raw_ids.split(";") if pid.strip()] if raw_ids else []
            golds[qid] = {
                "gold_answer": r.get("gold_answer", ""),
                "passage_ids": passage_ids,
            }
    return golds
