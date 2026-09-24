from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingest.extract import extract_html_sections

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

files = [
    (Path("data/raw/home_affairs_student_visa_work_hours_20230831_current.html"), "Current (2023-08-31)"),
    (Path("data/raw/home_affairs_student_visa_work_hours_20220607_superseded.html"), "Superseded (2022-06-07)")
]

terms = ["student visa", "hours", "fortnight", "48 hours", "40 hours", "work", "relaxation", "course", "8105"]

for path, label in files:
    sections = extract_html_sections(path, "TEST", 35)
    full_text = "\n\n".join(s["text"] for s in sections)
    print(f"=== {label} ===")
    print(f"File: {path.name}")
    print(f"Total extracted characters: {len(full_text)}")
    print(f"Total words: {len(full_text.split())}")
    print(f"Sections count: {len(sections)}")
    print(f"Exceeds 1,000 characters: {len(full_text) > 1000}")
    print("Term counts:")
    for t in terms:
        c = full_text.lower().count(t.lower())
        print(f'  - "{t}": {c}')
    print("\nFirst 500 characters of clean prose:")
    print("-" * 50)
    print(full_text[:500])
    print("-" * 50)
    print()
