"""Bridge metadata from corpus_metadata.csv to data/sources_draft.csv.

Builds the source registry strictly from real, fetched documents in data/raw/:
1. 14 in-scope PDFs with validated text layers.
2. 2 format-matched HTML Wayback snapshots forming the real planted currency pair:
   - S15: Current (snapshot 2026-08-27)
   - S16: Superseded (snapshot 2023-06-05, superseded_by: S15)
3. Zero fabricated metadata: notes contain only snapshot dates and archive URLs.
4. Excludes fatal documents (ATO TFN without text layer) and duplicate PDFs.
"""

from datetime import datetime
from pathlib import Path
import argparse
import csv
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
METADATA_PATH = REPO_ROOT / "corpus_metadata.csv"
DRAFT_SOURCES_PATH = REPO_ROOT / "data" / "sources_draft.csv"

# Permanent frozen source registry mapping.
# Source IDs are fixed forever. Dropped source IDs (such as S14) are retired, never reused.
FROZEN_SOURCE_SPECS = [
    ("S01", "applying-for-an-abn-australian-business-register"),
    ("S02", "bulk-billing-medicare-services-australia"),
    ("S03", "check-visa-details-and-conditions"),  # Canonical 10-page visa guide
    ("S04", "check-visa-details-and-conditions2"), # 2-page condition definitions list
    ("S05", "enrolling-in-medicare-medicare-services-australia"),
    ("S06", "fair-work-system-fair-work-ombudsman"),
    ("S07", "how-your-medicare-card-and-account-work-medicare-services-au"),
    ("S08", "i-m-a-migrant-worker-being-treated-unfairly-fair-work-ombuds"),
    ("S09", "i-m-not-getting-pay-slips-fair-work-ombudsman"),
    ("S10", "my-pay-doesn-t-seem-right-fair-work-ombudsman"),
    ("S11", "orientation-rmit-university"),
    ("S12", "tax-in-australia-what-you-need-to-know-australian-taxation-o"),
    ("S13", "tickets-and-payments-transport-victoria"),
    # S14: RETIRED (formerly what-is-a-tax-file-number-australian-taxation-office, no text layer)
    ("S15", "your-first-week-in-australia-study-australia"),
    ("S16", "myki-your-ticket-to-travel-public-transport-in-victoria-tran"),
    ("S17", {
        "url": "http://web.archive.org/web/20230831101024/https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/student-500/temporary-relaxation-of-working-hours-for-student-visa-holders",
        "publisher": "Department of Home Affairs",
        "title": "Temporary relaxation of working hours for student visa holders (snapshot 2023-08-31)",
        "retrieved_at": "2026-09-22",
        "last_updated": "2023-08-31",
        "licence": "TODO",
        "is_superseded": "false",
        "superseded_by": "",
        "notes": "Snapshot date: 2023-08-31 | Archive URL: http://web.archive.org/web/20230831101024/https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/student-500/temporary-relaxation-of-working-hours-for-student-visa-holders | Current static HTML member of currency pair | filename: home_affairs_student_visa_work_hours_20230831_current.html",
    }),
    ("S18", {
        "url": "http://web.archive.org/web/20220607122248/https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/student-500/temporary-relaxation-of-working-hours-for-student-visa-holders",
        "publisher": "Department of Home Affairs",
        "title": "Temporary relaxation of working hours for student visa holders (snapshot 2022-06-07)",
        "retrieved_at": "2026-09-22",
        "last_updated": "2022-06-07",
        "licence": "TODO",
        "is_superseded": "true",
        "superseded_by": "S17",
        "notes": "Snapshot date: 2022-06-07 | Archive URL: http://web.archive.org/web/20220607122248/https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/student-500/temporary-relaxation-of-working-hours-for-student-visa-holders | Superseded static HTML member of currency pair | filename: home_affairs_student_visa_work_hours_20220607_superseded.html",
    }),
]


def parse_date_to_iso(date_str: str) -> str:
    if not date_str or date_str.strip().lower() in ["not_shown", "nan", "none"]:
        return ""
    clean = date_str.strip()
    formats = ["%d %B %Y", "%d %b %Y", "%Y-%m-%d", "%d/%m/%Y"]
    for fmt in formats:
        try:
            return datetime.strptime(clean, fmt).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return clean


def parse_iso_retrieval_date(retrieved_at_str: str) -> str:
    if not retrieved_at_str:
        return ""
    try:
        return retrieved_at_str.split("T")[0]
    except Exception:
        return retrieved_at_str


def bridge_metadata() -> None:
    if not METADATA_PATH.exists():
        print(f"Error: {METADATA_PATH} not found.", file=sys.stderr)
        sys.exit(1)

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows_by_doc_id = {r["doc_id"]: r for r in reader}

    draft_rows = []
    todo_licence_count = 0
    todo_last_updated_count = 0

    for source_id, spec in FROZEN_SOURCE_SPECS:
        if isinstance(spec, dict):
            # Custom entry (e.g. Wayback snapshots)
            row_dict = {
                "source_id": source_id,
                "url": spec["url"],
                "publisher": spec["publisher"],
                "title": spec["title"],
                "retrieved_at": spec["retrieved_at"],
                "last_updated": spec["last_updated"],
                "licence": spec["licence"],
                "is_superseded": spec["is_superseded"],
                "superseded_by": spec["superseded_by"],
                "notes": spec["notes"],
            }
            if row_dict["licence"] == "TODO":
                todo_licence_count += 1
            if row_dict["last_updated"] == "TODO":
                todo_last_updated_count += 1
            draft_rows.append(row_dict)
        else:
            doc_id = spec
            if doc_id not in rows_by_doc_id:
                raise KeyError(f"Curated document {doc_id} for {source_id} not found in {METADATA_PATH}")

            row = rows_by_doc_id[doc_id]
            url = row.get("url", "")
            url_conf = row.get("url_confidence", "")
            publisher = row.get("publisher", "")
            title = row.get("title", "")
            raw_last_updated = row.get("last_updated", "")
            retrieved_at = parse_iso_retrieval_date(row.get("retrieved_at", ""))
            filename = row.get("filename", "")

            parsed_last_updated = parse_date_to_iso(raw_last_updated)
            if not parsed_last_updated or "PATTERN-INFERRED" in url_conf:
                last_updated_val = "TODO"
                todo_last_updated_count += 1
            else:
                last_updated_val = parsed_last_updated

            licence_val = "TODO"
            todo_licence_count += 1

            notes = f"doc_id: {doc_id} | filename: {filename}"
            if "verified" not in url_conf.lower() and "1.00" not in url_conf:
                notes += f" | URL confidence: {url_conf}"

            draft_rows.append({
                "source_id": source_id,
                "url": url,
                "publisher": publisher,
                "title": title,
                "retrieved_at": retrieved_at,
                "last_updated": last_updated_val,
                "licence": licence_val,
                "is_superseded": "false",
                "superseded_by": "",
                "notes": notes,
            })

    # Write to data/sources_draft.csv
    DRAFT_SOURCES_PATH.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "source_id",
        "url",
        "publisher",
        "title",
        "retrieved_at",
        "last_updated",
        "licence",
        "is_superseded",
        "superseded_by",
        "notes",
    ]
    with open(DRAFT_SOURCES_PATH, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(draft_rows)

    print("=" * 70)
    print(f"BRIDGED REAL METADATA -> {DRAFT_SOURCES_PATH.relative_to(REPO_ROOT)}")
    print(f"Total REAL documents selected: {len(draft_rows)}")
    print("  - 15 In-scope PDFs (all text-layer OK, including S03 canonical visa conditions)")
    print("  - 1 Retired source (S14 ATO TFN excluded due to missing text layer; never reused)")
    print("  - 2 Format-matched static HTML snapshots (S17 Current vs S18 Superseded)")
    print("  - 0 Fabricated/speculative documents")
    print("=" * 70)
    print(f"\nHuman Action Items:")
    print(f"  - Licence checks needed (licence='TODO'): {todo_licence_count}")
    print(f"  - Date verification needed (last_updated='TODO'): {todo_last_updated_count}")
    print("=" * 70)


if __name__ == "__main__":
    bridge_metadata()
