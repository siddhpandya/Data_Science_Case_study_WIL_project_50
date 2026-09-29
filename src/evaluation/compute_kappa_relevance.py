"""Compute Cohen's kappa between a filled relevance judging sheet and qrels.txt.

Usage:
    python src/evaluation/compute_kappa_relevance.py \\
        results/coding_sheets/relevance_judging_sheet.csv
"""

import csv
import sys
from pathlib import Path
from sklearn.metrics import cohen_kappa_score

REPO = Path(__file__).resolve().parent.parent.parent


def load_qrels():
    """Load qrels.txt into a dict of (qid, pid) -> grade."""
    qrels = {}
    with open(REPO / "data" / "qrels.txt", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                qid, _, pid, grade = parts[:4]
                qrels[(qid, pid)] = int(grade)
    return qrels


def main():
    if len(sys.argv) < 2:
        print("Usage: python compute_kappa_relevance.py <filled_judging_sheet.csv>")
        sys.exit(1)

    sheet_path = sys.argv[1]
    qrels = load_qrels()

    with open(sheet_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    human_labels = []
    qrels_labels = []

    for r in rows:
        label = r.get("label", "").strip()
        if not label:  # Skip unjudged
            continue

        qid = r["question_id"]
        pid = r["passage_id"]
        human_label = int(label)

        # Get qrels label (0 if not in qrels)
        qrels_label = qrels.get((qid, pid), 0)

        human_labels.append(human_label)
        qrels_labels.append(qrels_label)

    if len(human_labels) < 2:
        print("Not enough labelled rows for kappa computation.")
        return

    kappa = cohen_kappa_score(human_labels, qrels_labels)
    agree = sum(1 for h, q in zip(human_labels, qrels_labels) if h == q)

    print(f"Relevance judging vs qrels.txt")
    print(f"  Judged: {len(human_labels)} passages")
    print(f"  Agreement: {agree}/{len(human_labels)} ({agree/len(human_labels):.1%})")
    print(f"  Cohen's kappa: {kappa:.4f}")


if __name__ == "__main__":
    main()
