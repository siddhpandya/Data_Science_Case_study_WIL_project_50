"""Build collection.jsonl from extracted interim JSON files.

Reads data/interim/{source_id}.json files and produces a single
data/collection.jsonl with one JSON object per line:
  {"id": "S01_001", "contents": "heading\\n\\ntext", "source_id": "S01", "heading": "..."}

Passage IDs are deterministic: {source_id}_{section_index:03d}
"""

from pathlib import Path
import csv
import json
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SOURCES_PATH = REPO_ROOT / "data" / "sources.csv"
DRAFT_SOURCES_PATH = REPO_ROOT / "data" / "sources_draft.csv"
INTERIM_DIR = REPO_ROOT / "data" / "interim"
COLLECTION_PATH = REPO_ROOT / "data" / "collection.jsonl"


def build_collection(use_draft: bool = False) -> None:
    source_file = DRAFT_SOURCES_PATH if use_draft or not SOURCES_PATH.exists() else SOURCES_PATH
    if not source_file.exists():
        print(f"Error: {source_file} not found.", file=sys.stderr)
        sys.exit(1)

    with open(source_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sources = list(reader)

    passages = []
    for row in sources:
        source_id = row.get("source_id", "").strip()
        if not source_id:
            continue

        interim_path = INTERIM_DIR / f"{source_id}.json"
        if not interim_path.exists():
            print(f"  [SKIP] {source_id}: no interim JSON found", file=sys.stderr)
            continue

        with open(interim_path, "r", encoding="utf-8") as f:
            sections = json.load(f)

        for idx, section in enumerate(sections):
            passage_id = f"{source_id}_{idx + 1:03d}"
            heading = section.get("heading", "")
            text = section.get("text", "")
            contents = f"{heading}\n\n{text}" if heading else text

            passages.append({
                "id": passage_id,
                "contents": contents,
                "source_id": source_id,
                "heading": heading,
                "title": row.get("title", ""),
                "url": row.get("url", ""),
                "publisher": row.get("publisher", ""),
            })

    # Write collection.jsonl
    with open(COLLECTION_PATH, "w", encoding="utf-8") as f:
        for p in passages:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Collection built: {len(passages)} passages written to {COLLECTION_PATH}")
    print(f"Sources processed: {len(set(p['source_id'] for p in passages))}")

    # Print summary
    from collections import Counter
    source_counts = Counter(p["source_id"] for p in passages)
    for sid, count in sorted(source_counts.items()):
        print(f"  {sid}: {count} passages")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Build collection.jsonl from interim extractions")
    parser.add_argument("--draft", action="store_true", help="Use sources_draft.csv")
    args = parser.parse_args()
    build_collection(use_draft=args.draft)
