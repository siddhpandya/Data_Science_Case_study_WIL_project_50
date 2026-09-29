"""Compute Cohen's kappa between two filled coding sheets.

Usage:
    python src/evaluation/compute_kappa_coding.py \\
        results/coding_sheets/coding_sheet_coder1.csv \\
        results/coding_sheets/coding_sheet_coder2.csv
"""

import csv
import sys
from sklearn.metrics import cohen_kappa_score


def main():
    if len(sys.argv) < 3:
        print("Usage: python compute_kappa_coding.py <coder1.csv> <coder2.csv>")
        sys.exit(1)

    path1, path2 = sys.argv[1], sys.argv[2]

    with open(path1, encoding="utf-8") as f:
        rows1 = list(csv.DictReader(f))
    with open(path2, encoding="utf-8") as f:
        rows2 = list(csv.DictReader(f))

    assert len(rows1) == len(rows2), f"Row count mismatch: {len(rows1)} vs {len(rows2)}"

    columns = ["correct", "supported", "applicable", "citation_valid", "fabricated_quote"]

    print(f"{'Column':20s} | {'Kappa':8s} | {'n':5s} | {'Agree':5s}")
    print(f"{'-'*20} | {'-'*8} | {'-'*5} | {'-'*5}")

    for col in columns:
        labels1 = []
        labels2 = []
        for r1, r2 in zip(rows1, rows2):
            v1 = r1.get(col, "").strip()
            v2 = r2.get(col, "").strip()
            if v1 and v2:  # Both coders labelled this cell
                labels1.append(v1)
                labels2.append(v2)

        if len(labels1) < 2:
            print(f"{col:20s} | {'N/A':8s} | {len(labels1):5d} | {'N/A':5s}")
            continue

        kappa = cohen_kappa_score(labels1, labels2)
        agree = sum(1 for a, b in zip(labels1, labels2) if a == b)
        print(f"{col:20s} | {kappa:8.4f} | {len(labels1):5d} | {agree:5d}")


if __name__ == "__main__":
    main()
