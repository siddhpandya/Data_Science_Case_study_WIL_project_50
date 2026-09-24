"""Extraction module for SettleIN pipeline.

Reads registered sources strictly from data/sources.csv (or data/sources_draft.csv).
Extracts text from raw PDF and HTML documents:
1. Strips archive toolbars, navigation, headers, footers, and scripts for HTML.
2. Extracts text using pypdf for PDFs, preserving section heading structure.
3. Enforces deterministic section merging based on chunking.min_section_words from config.yaml,
   guaranteeing a stable, frozen unit of retrieval and relevance judgment across runs.
4. FAILS LOUDLY if any document yields near-zero characters (< 50 chars),
   preventing silent empty passages from entering the collection.
5. Emits intermediate structured JSON to data/interim/{source_id}.json
   as a list of {"heading": str, "level": int, "text": str}.
6. Emits candidate metadata to data/interim/{source_id}_meta.json for human review.
"""

from pathlib import Path
import argparse
import csv
import json
import re
import sys
import yaml
from bs4 import BeautifulSoup
from pypdf import PdfReader

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
SOURCES_PATH = REPO_ROOT / "data" / "sources.csv"
DRAFT_SOURCES_PATH = REPO_ROOT / "data" / "sources_draft.csv"
RAW_DIR = REPO_ROOT / "data" / "raw"
INTERIM_DIR = REPO_ROOT / "data" / "interim"

MIN_EXTRACTED_CHARS = 50


def load_min_section_words() -> int:
    """Load min_section_words from config/config.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("chunking", {}).get("min_section_words", 35)
    return 35


def find_raw_file_for_source(source_id: str, row: dict) -> Path:
    """Find the raw file corresponding to a source entry, strictly driven by metadata."""
    # 1. Direct source_id naming: data/raw/{source_id}.pdf or .html
    for ext in [".pdf", ".html", ".htm"]:
        candidate = RAW_DIR / f"{source_id}{ext}"
        if candidate.exists():
            return candidate

    # 2. Check filename recorded in notes: e.g. "filename: Check visa details and conditions.pdf"
    notes = row.get("notes", "")
    for part in notes.split("|"):
        if "filename:" in part:
            fname = part.split("filename:")[1].strip()
            candidate = RAW_DIR / fname
            if candidate.exists():
                return candidate

    raise FileNotFoundError(
        f"Raw file for {source_id} not found in {RAW_DIR}. "
        f"Expected {source_id}.pdf/.html or filename from notes."
    )


def merge_short_sections(sections: list[dict], min_words: int) -> list[dict]:
    """Deterministically merge short sub-sections into their parent/adjacent section.

    Ensures every passage meets the configured chunking.min_section_words threshold
    so the qrels judgment units are frozen and well-formed.
    """
    if not sections or min_words <= 0:
        return sections

    merged = []
    buffer = None

    for s in sections:
        if buffer is None:
            buffer = dict(s)
        elif len(buffer["text"].split()) < min_words:
            # Merge current section into buffer
            buffer["text"] += f"\n\n### {s['heading']}\n{s['text']}"
        else:
            merged.append(buffer)
            buffer = dict(s)

    # Check final buffer
    if buffer is not None:
        if merged and len(buffer["text"].split()) < min_words:
            merged[-1]["text"] += f"\n\n### {buffer['heading']}\n{buffer['text']}"
        else:
            merged.append(buffer)

    return merged


def extract_pdf_sections(pdf_path: Path, source_id: str, min_words: int) -> list[dict]:
    """Extract text from PDF using pypdf, detecting headings and failing loudly on empty layers."""
    try:
        reader = PdfReader(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF {pdf_path.name} for {source_id}: {e}")

    num_pages = len(reader.pages)
    if num_pages == 0:
        raise ValueError(f"[{source_id}] PDF {pdf_path.name} has 0 pages.")

    sections = []
    total_chars = 0
    full_text_chunks = []

    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        total_chars += len(page_text.strip())
        if page_text.strip():
            full_text_chunks.append((page_idx, page_text))

    # CRITICAL ACCEPTANCE CHECK: Fail loudly on empty or scanned PDFs with no text layer
    if total_chars < MIN_EXTRACTED_CHARS:
        raise ValueError(
            f"FATAL: Extraction failed for [{source_id}] ({pdf_path.name}). "
            f"Extracted only {total_chars} characters across {num_pages} page(s). "
            f"Document has NO valid text layer (must be re-saved from Chrome with selectable text)."
        )

    # Heuristic section extraction based on page blocks and headings
    for page_idx, page_text in full_text_chunks:
        lines = [line.strip() for line in page_text.split("\n") if line.strip()]
        current_heading = f"Page {page_idx}"
        current_level = 1
        current_paras = []

        for line in lines:
            if (len(line) < 80 and not line.endswith(".") and
                    (line.isupper() or line.istitle() or re.match(r"^[A-Z0-9\.\-\s]{3,60}$", line))):
                if current_paras:
                    section_text = " ".join(current_paras).strip()
                    if len(section_text) >= 20:
                        sections.append({
                            "heading": current_heading,
                            "level": current_level,
                            "text": section_text,
                        })
                    current_paras = []
                current_heading = line
                current_level = 2
            else:
                current_paras.append(line)

        if current_paras:
            section_text = " ".join(current_paras).strip()
            if len(section_text) >= 20:
                sections.append({
                    "heading": current_heading,
                    "level": current_level,
                    "text": section_text,
                })

    if not sections:
        all_text = " ".join([chunk[1].strip() for chunk in full_text_chunks])
        sections.append({
            "heading": pdf_path.stem,
            "level": 1,
            "text": all_text,
        })

    return merge_short_sections(sections, min_words)


def extract_html_sections(html_path: Path, source_id: str, min_words: int) -> list[dict]:
    """Extract structured sections from HTML, stripping archive chrome and boilerplate."""
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")

    # 1. Strip Wayback toolbar / archive-injected elements
    for el in soup.find_all(id=re.compile(r"wm-ipp|wm-capinfo|wm-share|donato", re.I)):
        el.decompose()
    for el in soup.find_all(class_=re.compile(r"wm-|wb-autocomplete", re.I)):
        el.decompose()

    # 2. Strip scripts, styles, noscript, nav, header, footer, svgs, aside
    for tag in soup(["script", "style", "nav", "noscript", "svg", "header", "footer", "aside"]):
        tag.decompose()

    # 3. Strip search widgets, breadcrumbs, sidebars, and navigation containers
    for el in soup.find_all(class_=re.compile(r"breadcrumb|search|navigation|nav-container|page-header|site-header|footer|side-nav|sidebar|sub-nav", re.I)):
        el.decompose()
    for el in soup.find_all(id=re.compile(r"breadcrumb|search|navigation|nav-container|header|footer|side-nav|sidebar|sub-nav", re.I)):
        el.decompose()

    # 4. Locate main content container (supporting SharePoint DeltaPlaceHolderMain)
    main_container = (
        soup.find(id="DeltaPlaceHolderMain") or
        soup.find("main") or
        soup.find("article") or
        soup.find(id="content") or
        soup.find(id="s4-workspace") or
        soup.body
    )

    if not main_container:
        raise ValueError(f"[{source_id}] Could not locate main body in {html_path.name}.")

    sections = []
    current_heading = html_path.stem
    current_level = 1
    current_paragraphs = []
    total_chars = 0

    skip_phrases = ["skip to main content", "skip to content", "popular searches", "your previous searches"]

    for elem in main_container.descendants:
        if elem.name in ["h1", "h2", "h3", "h4"]:
            heading_text = elem.get_text().replace("\u200b", "").replace("\xa0", " ").strip()
            if heading_text:
                if current_paragraphs:
                    body = " ".join(current_paragraphs).strip()
                    if len(body) >= 20:
                        sections.append({
                            "heading": current_heading,
                            "level": current_level,
                            "text": body,
                        })
                    current_paragraphs = []
                current_heading = heading_text
                current_level = int(elem.name[1])
        elif elem.name in ["p", "li"]:
            text = elem.get_text().replace("\u200b", "").replace("\xa0", " ").strip()
            if text and len(text) > 10:
                if any(sp in text.lower() for sp in skip_phrases):
                    continue
                total_chars += len(text)
                current_paragraphs.append(text)

    if current_paragraphs:
        body = " ".join(current_paragraphs).strip()
        if len(body) >= 20:
            sections.append({
                "heading": current_heading,
                "level": current_level,
                "text": body,
            })

    # If no heading splits occurred, wrap text as a single section
    if not sections and total_chars >= MIN_EXTRACTED_CHARS:
        all_text = " ".join(current_paragraphs).strip()
        sections.append({
            "heading": html_path.stem,
            "level": 1,
            "text": all_text,
        })

    if total_chars < MIN_EXTRACTED_CHARS:
        raise ValueError(
            f"FATAL: Extraction failed for [{source_id}] ({html_path.name}). "
            f"Extracted only {total_chars} characters. File appears empty or unrendered."
        )

    return merge_short_sections(sections, min_words)


def run_extraction(use_draft: bool = False) -> None:
    source_file = DRAFT_SOURCES_PATH if use_draft or not SOURCES_PATH.exists() else SOURCES_PATH
    if not source_file.exists():
        print(f"Error: {source_file} not found.", file=sys.stderr)
        sys.exit(1)

    min_words = load_min_section_words()
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    with open(source_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        sources = list(reader)

    print(f"Running extraction for {len(sources)} sources defined in {source_file.name} (min_section_words={min_words})...")
    success_count = 0
    total_sections_count = 0
    failed_sources = []

    for row in sources:
        source_id = row.get("source_id", "")
        if not source_id:
            continue

        try:
            raw_path = find_raw_file_for_source(source_id, row)
        except FileNotFoundError as e:
            failed_sources.append((source_id, str(e)))
            continue

        try:
            if raw_path.suffix.lower() == ".pdf":
                sections = extract_pdf_sections(raw_path, source_id, min_words)
            elif raw_path.suffix.lower() in [".html", ".htm"]:
                sections = extract_html_sections(raw_path, source_id, min_words)
            else:
                raise ValueError(f"Unsupported file format: {raw_path.suffix}")

            # Write intermediate JSON
            interim_json = INTERIM_DIR / f"{source_id}.json"
            with open(interim_json, "w", encoding="utf-8") as out:
                json.dump(sections, out, indent=2, ensure_ascii=False)

            # Write candidate metadata
            meta_json = INTERIM_DIR / f"{source_id}_meta.json"
            meta_data = {
                "source_id": source_id,
                "raw_file": raw_path.name,
                "section_count": len(sections),
                "total_characters": sum(len(s["text"]) for s in sections),
                "url": row.get("url", ""),
                "publisher": row.get("publisher", ""),
                "title": row.get("title", ""),
                "detected_date": row.get("last_updated", ""),
            }
            with open(meta_json, "w", encoding="utf-8") as out:
                json.dump(meta_data, out, indent=2)

            success_count += 1
            total_sections_count += len(sections)
            print(f"  [OK] {source_id}: {len(sections)} sections, {meta_data['total_characters']} chars ({raw_path.name})")

        except Exception as e:
            failed_sources.append((source_id, str(e)))
            print(f"  [FAIL] {source_id}: {e}", file=sys.stderr)

    print("\n" + "=" * 60)
    print(f"EXTRACTION COMPLETE: {success_count} sources succeeded, {len(failed_sources)} failed.")
    print(f"TOTAL SECTION PASSAGES PRODUCED: {total_sections_count}")
    print("=" * 60)
    if failed_sources:
        print("\nFailures:")
        for s_id, err in failed_sources:
            print(f"  - [{s_id}] {err}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from registered raw documents")
    parser.add_argument("--draft", action="store_true", help="Extract using data/sources_draft.csv")
    args = parser.parse_args()
    run_extraction(use_draft=args.draft)
