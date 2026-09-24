import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

from src.ingest.extract import extract_html_sections

for p in [
    Path("data/raw/home_affairs_student_visa_work_hours_20230831_current.html"),
    Path("data/raw/home_affairs_student_visa_work_hours_20220607_superseded.html")
]:
    secs = extract_html_sections(p, p.stem, 35)
    print(f"=== {p.name} ===")
    print(f"Total sections: {len(secs)}")
    for i, s in enumerate(secs):
        print(f"  Section {i+1}: Heading=\"{s['heading']}\", Words={len(s['text'].split())}, Chars={len(s['text'])}")
    print()
