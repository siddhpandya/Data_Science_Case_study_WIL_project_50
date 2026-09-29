"""Generate currency coding sheets for the 6 currency answers.

Same columns as the main coding sheet, plus states_outdated_rule column.
Shuffle, hide which retriever produced each answer.
"""

import csv
import json
import random
import re
from pathlib import Path

REPO = Path(r"c:\Users\Abhishek\OneDrive\Documents\Data_Science_Case_study_WIL_project_50")


def load_gold():
    with open(REPO / "data" / "gold_answers.csv", encoding="utf-8") as f:
        return {r["question_id"].strip(): r for r in csv.DictReader(f)}


def _normalise(text):
    """Lowercase, straighten quotes/apostrophes, collapse whitespace."""
    t = text.lower()
    t = t.replace("\u2018", "'").replace("\u2019", "'")
    t = t.replace("\u201c", '"').replace("\u201d", '"')
    t = re.sub(r"\s+", " ", t).strip()
    return t


def has_fabricated_quote(answer, passages_text):
    """Check if answer contains a 20+ char quoted string not in retrieved passages."""
    norm_passages = _normalise(passages_text)
    straight = re.findall(r'"([^"]{20,})"', answer)
    curly = re.findall(r'\u201c([^\u201d]{20,})\u201d', answer)
    quotes = straight + curly
    for q in quotes:
        if _normalise(q) not in norm_passages:
            return True
    return False


def load_collection():
    passages = {}
    with open(REPO / "data" / "collection.jsonl", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line)
            passages[p["id"]] = p["contents"]
    return passages


def main():
    gold = load_gold()
    collection = load_collection()

    rows = []
    key_rows = []

    for fname, label in [
        ("currency_with_superseded.jsonl", "dense_currency"),
        ("currency_bm25_with_superseded.jsonl", "bm25_currency"),
    ]:
        recs = [json.loads(l) for l in open(REPO / "results" / "generations" / fname, encoding="utf-8")]
        for r in recs:
            qid = r["question_id"]
            g = gold.get(qid, {})
            retrieved_ids = r.get("retrieved_ids", [])
            # Build passages text for fabricated_quote check
            passages_text = " ".join(collection.get(pid, "") for pid in retrieved_ids)

            rows.append({
                "_source": label,
                "_question_id": qid,
                "question": r["question"],
                "answer": r["answer"],
                "retrieved_passages": "; ".join(retrieved_ids),
                "gold_answer": g.get("gold_answer", ""),
                "correct": "",
                "supported": "",
                "applicable": "",
                "citation_valid": "",
                "fabricated_quote": "YES" if has_fabricated_quote(r["answer"], passages_text) else "",
                "states_outdated_rule": "",
                "notes": "",
            })

    # Shuffle
    random.seed(42)
    random.shuffle(rows)

    # Extract key
    key_rows = [{"row_number": i+1, "question_id": r["_question_id"], "source": r["_source"]}
                for i, r in enumerate(rows)]

    # Strip hidden fields
    public_fields = [k for k in rows[0].keys() if not k.startswith("_")]
    public_rows = [{k: r[k] for k in public_fields} for r in rows]

    out_dir = REPO / "results" / "coding_sheets"
    out_dir.mkdir(parents=True, exist_ok=True)

    for coder in ["coder1", "coder2"]:
        path = out_dir / f"currency_coding_sheet_{coder}.csv"
        with open(path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=public_fields)
            writer.writeheader()
            writer.writerows(public_rows)
        print(f"  {path} ({len(public_rows)} rows)")

    key_path = out_dir / "currency_coding_sheet_key.csv"
    with open(key_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(key_rows[0].keys()))
        writer.writeheader()
        writer.writerows(key_rows)
    print(f"  Key: {key_path}")


if __name__ == "__main__":
    main()
