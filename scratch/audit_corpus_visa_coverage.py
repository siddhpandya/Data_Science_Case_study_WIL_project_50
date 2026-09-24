from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.ingest.extract import extract_pdf_sections, extract_html_sections, load_min_section_words

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

raw_dir = REPO_ROOT / "data" / "raw"
terms = ["8105", "fortnight", "work condition", "student visa"]
min_words = load_min_section_words()

print("=" * 80)
print(f"CORPUS VISA COVERAGE AUDIT (data/raw/, min_section_words={min_words}):")
print("=" * 80)

total_passages = 0
matches_by_term = {t: 0 for t in terms}
source_reports = []

for p in sorted(raw_dir.glob("*")):
    if p.name.startswith("."):
        continue
    
    if p.suffix.lower() == ".pdf":
        sections = extract_pdf_sections(p, p.stem, min_words)
    elif p.suffix.lower() in [".html", ".htm"]:
        sections = extract_html_sections(p, p.stem, min_words)
    else:
        continue
        
    num_passages = len(sections)
    total_passages += num_passages
    
    file_term_counts = {t: 0 for t in terms}
    passages_with_terms = {t: 0 for t in terms}
    
    for s in sections:
        sec_text = (s["heading"] + " " + s["text"]).lower()
        for t in terms:
            c = sec_text.count(t)
            if c > 0:
                file_term_counts[t] += c
                passages_with_terms[t] += 1
                
    for t in terms:
        matches_by_term[t] += file_term_counts[t]
        
    source_reports.append({
        "file": p.name,
        "passages": num_passages,
        "term_counts": file_term_counts,
        "passages_matching": passages_with_terms
    })

print(f"\nTotal files processed: {len(source_reports)}")
print(f"Total section passages generated: {total_passages}")
print("\nAggregate Occurrences Across Corpus:")
for t in terms:
    print(f'  - "{t}": {matches_by_term[t]} total occurrences')

print("\nPer-Source Breakdown:")
print("-" * 80)
for rep in source_reports:
    has_any = any(rep["term_counts"].values())
    mark = "[MATCH]" if has_any else "[     ]"
    print(f"{mark} {rep['file']} ({rep['passages']} passages):")
    if has_any:
        for t in terms:
            cnt = rep["term_counts"][t]
            p_cnt = rep["passages_matching"][t]
            if cnt > 0:
                print(f"       \"{t}\": {cnt} occurrences across {p_cnt} passage(s)")
    else:
        print("       (No visa terms matched)")
print("-" * 80)
