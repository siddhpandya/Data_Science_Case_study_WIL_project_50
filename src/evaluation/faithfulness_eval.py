"""Faithfulness evaluation — manual coding sheets only.

This module does NOT call the LLM. It generates coding sheets for two
human coders to independently label each claim as SUPPORTED / PARTIAL /
UNSUPPORTED, and a script to compute Cohen's kappa once both are filled.

Output:
  results/faithfulness_coding_sheet_A.csv
  results/faithfulness_coding_sheet_B.csv
"""

import csv
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def split_sentences(text: str) -> list[str]:
    """Split text into sentences. Simple regex-based."""
    # Remove citation brackets like [S03_002]
    text = re.sub(r"\[S\d{2}_\d{3}\]", "", text)
    text = re.sub(r"\[[\w]+\]", "", text)
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s.strip() for s in sentences if len(s.strip()) > 15]


def load_collection_map() -> dict[str, dict]:
    """Load collection → {passage_id: passage}."""
    col_path = REPO_ROOT / "data" / "collection.jsonl"
    passages = {}
    with open(col_path, "r", encoding="utf-8") as f:
        for line in f:
            p = json.loads(line.strip())
            passages[p["id"]] = p
    return passages


def generate_coding_sheets(config_name: str) -> None:
    """Generate faithfulness coding sheets for a single configuration."""
    gen_path = REPO_ROOT / "results" / "generations" / f"{config_name}.jsonl"
    if not gen_path.exists():
        print(f"  No generations found: {gen_path}")
        return

    generations = []
    with open(gen_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                generations.append(json.loads(line.strip()))

    collection = load_collection_map()
    rows = []

    for gen in generations:
        if gen.get("refused_strict", gen.get("refused", False)):
            continue  # Skip refusals

        answer = gen["answer"]
        sentences = split_sentences(answer)
        retrieved_ids = gen.get("retrieved_ids", [])

        # Build passages text for the coder's reference
        passages_text = ""
        for pid in retrieved_ids:
            p = collection.get(pid)
            if p:
                passages_text += f"[{pid}]: {p['contents'][:500]}\n\n"

        for i, sentence in enumerate(sentences, 1):
            rows.append({
                "question_id": gen["question_id"],
                "config": config_name,
                "sentence_idx": i,
                "sentence": sentence,
                "passages_summary": passages_text[:1000],
                "label": "",  # SUPPORTED / PARTIAL / UNSUPPORTED
                "notes": "",
            })

    if not rows:
        print(f"  No sentences to code for {config_name}")
        return

    # Write two blank copies
    for coder in ["A", "B"]:
        out_path = REPO_ROOT / "results" / f"faithfulness_coding_sheet_{coder}.csv"
        fields = list(rows[0].keys())
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Written: {out_path} ({len(rows)} sentences)")


def compute_kappa():
    """Compute Cohen's kappa between the two filled coding sheets."""
    sheet_a_path = REPO_ROOT / "results" / "faithfulness_coding_sheet_A.csv"
    sheet_b_path = REPO_ROOT / "results" / "faithfulness_coding_sheet_B.csv"

    if not sheet_a_path.exists() or not sheet_b_path.exists():
        print("Both coding sheets must be filled before computing kappa.")
        return

    with open(sheet_a_path, "r", encoding="utf-8") as f:
        rows_a = list(csv.DictReader(f))
    with open(sheet_b_path, "r", encoding="utf-8") as f:
        rows_b = list(csv.DictReader(f))

    if len(rows_a) != len(rows_b):
        print(f"Sheet length mismatch: A={len(rows_a)}, B={len(rows_b)}")
        return

    labels_a = [r["label"].strip().upper() for r in rows_a]
    labels_b = [r["label"].strip().upper() for r in rows_b]

    # Check all labels are filled
    empty_a = sum(1 for l in labels_a if not l)
    empty_b = sum(1 for l in labels_b if not l)
    if empty_a or empty_b:
        print(f"Unfilled labels: sheet A has {empty_a}, sheet B has {empty_b}")
        return

    # Cohen's kappa
    categories = ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]
    n = len(labels_a)

    # Observed agreement
    agree = sum(1 for a, b in zip(labels_a, labels_b) if a == b)
    po = agree / n

    # Expected agreement
    pe = 0.0
    for cat in categories:
        count_a = sum(1 for l in labels_a if l == cat)
        count_b = sum(1 for l in labels_b if l == cat)
        pe += (count_a / n) * (count_b / n)

    kappa = (po - pe) / (1 - pe) if pe < 1 else 0.0

    print(f"\nCohen's kappa: {kappa:.4f}")
    print(f"  Observed agreement: {po:.4f} ({agree}/{n})")
    print(f"  Expected agreement: {pe:.4f}")
    print(f"  Interpretation: ", end="")
    if kappa < 0:
        print("poor")
    elif kappa < 0.20:
        print("slight")
    elif kappa < 0.40:
        print("fair")
    elif kappa < 0.60:
        print("moderate")
    elif kappa < 0.80:
        print("substantial")
    else:
        print("almost perfect")


def generate_all_sheets():
    """Generate coding sheets for all generation configs."""
    gen_dir = REPO_ROOT / "results" / "generations"
    if not gen_dir.exists():
        print("No generations directory found.")
        return

    for gen_file in sorted(gen_dir.glob("*.jsonl")):
        config_name = gen_file.stem
        print(f"Generating coding sheets: {config_name}")
        generate_coding_sheets(config_name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--kappa", action="store_true", help="Compute kappa from filled sheets")
    args = parser.parse_args()

    if args.kappa:
        compute_kappa()
    else:
        generate_all_sheets()
