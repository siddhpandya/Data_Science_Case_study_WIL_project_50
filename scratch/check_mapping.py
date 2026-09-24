import csv
from pathlib import Path

REPO_ROOT = Path(".").resolve()
METADATA_PATH = REPO_ROOT / "corpus_metadata.csv"

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows_by_id = {r["doc_id"]: r for r in reader}

FROZEN_MAPPING = [
    ("S01", "applying-for-an-abn-australian-business-register"),
    ("S02", "bulk-billing-medicare-services-australia"),
    ("S03", "check-visa-details-and-conditions"),
    ("S04", "check-visa-details-and-conditions2"),
    ("S05", "enrolling-in-medicare-medicare-services-australia"),
    ("S06", "fair-work-system-fair-work-ombudsman"),
    ("S07", "how-your-medicare-card-and-account-work-medicare-services-au"),
    ("S08", "i-m-a-migrant-worker-being-treated-unfairly-fair-work-ombuds"),
    ("S09", "i-m-not-getting-pay-slips-fair-work-ombudsman"),
    ("S10", "my-pay-doesn-t-seem-right-fair-work-ombudsman"),
    ("S11", "orientation-rmit-university"),
    ("S12", "tax-in-australia-what-you-need-to-know-australian-taxation-o"),
    ("S13", "tickets-and-payments-transport-victoria"),
    # S14 is RETIRED
    ("S15", "your-first-week-in-australia-study-australia"),
    ("S16", "myki-your-ticket-to-travel-public-transport-in-victoria-tran"),
]

for sid, doc_id in FROZEN_MAPPING:
    if doc_id not in rows_by_id:
        print(f"ERROR: {doc_id} for {sid} NOT found in corpus_metadata.csv")
    else:
        print(f"{sid}: {doc_id} -> {rows_by_id[doc_id]['filename']}")
