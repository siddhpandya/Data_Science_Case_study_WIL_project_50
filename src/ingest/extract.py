"""Extraction module for SettleIN pipeline.

Reads registered sources strictly from data/sources.csv (or data/sources_draft.csv).
Extracts text from raw PDF and HTML documents:
1. Strips archive toolbars, navigation, headers, footers, and scripts for HTML.
2. Extracts text using PyMuPDF (fitz) for PDFs, with font-size heading detection.
3. Strips Chrome print headers/footers, boilerplate disclaimers, and QC markers
   BEFORE sectioning and character counting.
4. Enforces deterministic section merging based on chunking.min_section_words from config.yaml,
   guaranteeing a stable, frozen unit of retrieval and relevance judgment across runs.
5. FAILS LOUDLY if:
   - any document yields < 50 chars AFTER boilerplate stripping, or
   - fewer than 20% of alphabetic words contain the letter 'i' or 'l' (garbling).
6. Emits intermediate structured JSON to data/interim/{source_id}.json
   as a list of {"heading": str, "level": int, "text": str}.
7. Emits candidate metadata to data/interim/{source_id}_meta.json for human review.
"""

from pathlib import Path
import argparse
import csv
import json
import re
import sys
import yaml
from bs4 import BeautifulSoup

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "config.yaml"
SOURCES_PATH = REPO_ROOT / "data" / "sources.csv"
DRAFT_SOURCES_PATH = REPO_ROOT / "data" / "sources_draft.csv"
RAW_DIR = REPO_ROOT / "data" / "raw"
INTERIM_DIR = REPO_ROOT / "data" / "interim"

MIN_EXTRACTED_CHARS = 50
MIN_IL_RATIO = 0.20  # Minimum fraction of words containing i or l

# ── Boilerplate patterns ────────────────────────────────────────────────────

# Chrome print header: "11/09/2026, 16:58 Title https://url.com page/total"
CHROME_HEADER_RE = re.compile(
    r"^\d{1,2}/\d{1,2}/\d{4},?\s+\d{1,2}[:.]\d{2}(?:\s*[ap]m)?\s+.*$",
    re.IGNORECASE
)

# Chrome print footer: "https://www.example.com/..." or "Page N of M" at start of line
CHROME_FOOTER_URL_RE = re.compile(
    r"^https?://\S+\s*$", re.IGNORECASE
)
CHROME_PAGE_FOOTER_RE = re.compile(
    r"^Page\s+\d+\s+of\s+\d+\s*$", re.IGNORECASE
)
# Combined "N/M" pattern at end of a line (e.g. "1/10")
CHROME_PAGE_FRAC_RE = re.compile(
    r"^\d{1,3}/\d{1,3}\s*$"
)

# Fair Work standard disclaimer
FAIR_WORK_DISCLAIMER_RE = re.compile(
    r"The Fair Work Ombudsman is committed to providing you with advice that you can rely on",
    re.IGNORECASE
)

# Services Australia "QC nnnnn" markers
QC_MARKER_RE = re.compile(r"^QC\s+\d{4,6}\s*$", re.IGNORECASE)

# "This information was printed..." block
PRINTED_INFO_RE = re.compile(
    r"^This information was printed", re.IGNORECASE
)

# "Printed link references" heading and subsequent bare URLs
PRINTED_LINK_REFS_RE = re.compile(
    r"^Printed link references\s*$", re.IGNORECASE
)

# Services Australia footer pattern: "Page last updated: ..." as standalone
SA_FOOTER_RE = re.compile(
    r"^(Page )?last (updated|modified):?\s+\d+", re.IGNORECASE
)

# Material icon text that leaks from web rendering
MATERIAL_ICON_RE = re.compile(
    r"^(keyboard_arrow_right|keyboard_arrow_left|keyboard_return|"
    r"chevron_right|chevron_left|expand_more|expand_less|east|west|"
    r"search|menu|close|arrow_forward|arrow_back|expand_circle_down)\s*$",
    re.IGNORECASE
)

# Patterns that should never be used as section headings
BAD_HEADING_PATTERNS = [
    re.compile(r"^Did you find what you were looking for", re.I),
    re.compile(r"Infoline\s+13\s+13\s+94", re.I),
    re.compile(r"^Work Infoline", re.I),
    re.compile(r"^Printed link references\s*$", re.I),
    re.compile(r"^On this page:?\s*$", re.I),
    re.compile(r"^on this page\s*$", re.I),
    re.compile(r"^Follow Us\s*$", re.I),
    re.compile(r"^Related information\s*$", re.I),
    re.compile(r"^Bookmark to My account", re.I),
    re.compile(r"^\d{1,2}/\d{1,2}/\d{4}"),  # Date strings
    re.compile(r"^https?://"),  # URLs
    re.compile(r"^\(.*\)\s*$"),  # Bare parenthetical references
    re.compile(r"^\(#\)\s*$"),  # Anchor-only links
    re.compile(r"^\[\d+\]\s*$"),  # Numbered references like [1]
    re.compile(r"^\\u[0-9a-fA-F]{4}"),  # Unicode escape sequences
]


def load_min_section_words() -> int:
    """Load min_section_words from config/config.yaml."""
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
            return cfg.get("chunking", {}).get("min_section_words", 35)
    return 35


def il_ratio(text: str) -> float:
    """Fraction of 3+ letter alphabetic words containing 'i' or 'l'.

    Normal English text has roughly 40% of words containing these letters.
    Garbled PDFs with broken font encoding drop well below 20%.
    """
    words = [w for w in re.findall(r"[a-zA-Z]+", text) if len(w) >= 3]
    if not words:
        return 0.0
    return sum(1 for w in words if "i" in w.lower() or "l" in w.lower()) / len(words)


def is_boilerplate_line(line: str) -> bool:
    """Check if a single line is boilerplate that should be stripped."""
    stripped = line.strip()
    if not stripped:
        return True
    if CHROME_HEADER_RE.match(stripped):
        return True
    if CHROME_FOOTER_URL_RE.match(stripped):
        return True
    if CHROME_PAGE_FOOTER_RE.match(stripped):
        return True
    if CHROME_PAGE_FRAC_RE.match(stripped):
        return True
    if QC_MARKER_RE.match(stripped):
        return True
    if PRINTED_INFO_RE.match(stripped):
        return True
    if PRINTED_LINK_REFS_RE.match(stripped):
        return True
    if SA_FOOTER_RE.match(stripped):
        return True
    if MATERIAL_ICON_RE.match(stripped):
        return True
    # Fair Work / Services Australia feedback and footer patterns
    if stripped.startswith("Did you find what you were looking for"):
        return True
    if stripped.startswith("Bookmark to My account"):
        return True
    if re.match(r"^Infoline\s+13\s+13\s+94", stripped, re.I):
        return True
    if re.match(r"^Work Infoline", stripped, re.I):
        return True
    # Bare anchor links like "(#)" or "(https://...)"
    if re.match(r"^\(#\)\s*$", stripped):
        return True
    if re.match(r"^\([a-z]+://[^)]+\)\s*$", stripped):
        return True
    # Single unicode characters (icon glyphs from web fonts)
    if len(stripped) <= 2 and not stripped.isascii():
        return True
    # Bare bracket references like "[1]", "[2]"
    if re.match(r"^\[\d+\]\s*$", stripped):
        return True
    return False


def strip_boilerplate_lines(lines: list[str]) -> list[str]:
    """Remove boilerplate lines and Fair Work disclaimer paragraphs."""
    cleaned = []
    in_disclaimer = False
    in_printed_refs = False

    for line in lines:
        stripped = line.strip()

        # Skip individual boilerplate lines
        if is_boilerplate_line(stripped):
            # Check if we're entering a printed link references block
            if PRINTED_LINK_REFS_RE.match(stripped):
                in_printed_refs = True
            continue

        # Inside printed link references block: skip bare URLs and short entries
        if in_printed_refs:
            if re.match(r"^\[?\d+\]?\s*https?://", stripped) or re.match(r"^https?://", stripped):
                continue
            elif len(stripped) < 10:
                continue
            else:
                in_printed_refs = False

        # Detect and skip Fair Work disclaimer paragraphs
        if FAIR_WORK_DISCLAIMER_RE.search(stripped):
            in_disclaimer = True
            continue
        if in_disclaimer:
            # The disclaimer is typically a multi-line paragraph; skip until we hit
            # a clearly new section (short line that looks like a heading or empty gap)
            if len(stripped) < 20 and not stripped.endswith("."):
                in_disclaimer = False
                cleaned.append(line)
            continue

        cleaned.append(line)

    return cleaned


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


def _compute_font_thresholds(doc) -> dict:
    """Analyse a PyMuPDF document to determine heading font-size thresholds.

    Returns a dict with:
      body_size: the most common font size (assumed body text)
      heading_sizes: list of (size, level) tuples, largest first
    """
    size_char_counts: dict[float, int] = {}

    for page in doc:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    sz = round(span["size"], 1)
                    text = span["text"].strip()
                    if text and not is_boilerplate_line(text):
                        size_char_counts[sz] = size_char_counts.get(sz, 0) + len(text)

    if not size_char_counts:
        return {"body_size": 10.0, "heading_sizes": []}

    # Body size = the font size with the most total characters
    body_size = max(size_char_counts, key=size_char_counts.get)

    # Heading sizes = font sizes meaningfully larger than body (>1pt gap),
    # sorted largest first
    heading_sizes_raw = sorted(
        [sz for sz in size_char_counts if sz > body_size + 1.0],
        reverse=True
    )

    # Assign heading levels: h1 for largest, h2 for next, etc.
    heading_sizes = []
    for i, sz in enumerate(heading_sizes_raw):
        level = min(i + 1, 4)  # Cap at h4
        heading_sizes.append((sz, level))

    return {"body_size": body_size, "heading_sizes": heading_sizes}


def extract_pdf_sections_fitz(pdf_path: Path, source_id: str, min_words: int) -> list[dict]:
    """Extract text from PDF using PyMuPDF with font-size heading detection.

    Key improvements over pypdf page-based extraction:
    - Headings detected by font size, not text heuristics
    - Page breaks never start a new passage
    - Boilerplate stripped before sectioning
    - Garbling detected and reported
    """
    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF {pdf_path.name} for {source_id}: {e}")

    num_pages = len(doc)
    if num_pages == 0:
        raise ValueError(f"[{source_id}] PDF {pdf_path.name} has 0 pages.")

    # Step 1: Compute font thresholds from the whole document
    font_info = _compute_font_thresholds(doc)
    body_size = font_info["body_size"]
    heading_sizes = font_info["heading_sizes"]  # List of (size, level)

    def get_heading_level(font_size: float, is_bold: bool) -> int | None:
        """Return heading level if font_size indicates a heading, else None."""
        for h_size, h_level in heading_sizes:
            if font_size >= h_size - 0.5:  # Allow 0.5pt tolerance
                return h_level
        # Bold text at body size or slightly above could be a sub-heading
        if is_bold and font_size >= body_size + 0.5:
            return min(len(heading_sizes) + 1, 4)
        return None

    # Step 2: Extract all spans across all pages (continuous, ignoring page breaks)
    all_spans = []
    for page in doc:
        blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE)["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                line_text_parts = []
                line_max_size = 0
                line_is_bold = False
                for span in line["spans"]:
                    text = span["text"]
                    sz = round(span["size"], 1)
                    bold = bool(span["flags"] & (1 << 4))
                    if text.strip():
                        line_text_parts.append(text)
                        if sz > line_max_size:
                            line_max_size = sz
                        if bold:
                            line_is_bold = True
                if line_text_parts:
                    joined = "".join(line_text_parts).strip()
                    if joined:
                        all_spans.append({
                            "text": joined,
                            "size": line_max_size,
                            "bold": line_is_bold,
                        })

    doc.close()

    # Step 3: Strip boilerplate
    clean_spans = []
    for span in all_spans:
        if not is_boilerplate_line(span["text"]):
            clean_spans.append(span)

    # Step 4: Check total content after boilerplate stripping
    total_clean_text = " ".join(s["text"] for s in clean_spans)
    total_clean_chars = len(total_clean_text.strip())

    if total_clean_chars < MIN_EXTRACTED_CHARS:
        raise ValueError(
            f"FATAL: Extraction failed for [{source_id}] ({pdf_path.name}). "
            f"After boilerplate stripping, only {total_clean_chars} characters remain "
            f"across {num_pages} page(s). Document has NO valid content."
        )

    # Step 5: Check for garbling
    ratio = il_ratio(total_clean_text)
    if ratio < MIN_IL_RATIO:
        raise ValueError(
            f"FATAL: Garbling detected for [{source_id}] ({pdf_path.name}). "
            f"i/l word ratio = {ratio:.3f} (minimum {MIN_IL_RATIO}). "
            f"The PDF text layer has broken font encoding. "
            f"Re-capture this document as HTML from its live URL."
        )

    # Step 6: Build sections using font-size heading detection
    sections = []
    current_heading = pdf_path.stem
    current_level = 1
    current_paras = []

    for span in clean_spans:
        hlevel = get_heading_level(span["size"], span["bold"])
        text = span["text"]

        # Classify as heading only if:
        # 1. Font size indicates heading level
        # 2. Text is short enough to be a heading (<80 chars)
        # 3. Doesn't end with a period (sentence, not heading)
        # 4. Doesn't match any bad heading pattern
        is_heading = (
            hlevel is not None
            and len(text) < 80
            and not text.endswith(".")
            and not any(p.search(text) for p in BAD_HEADING_PATTERNS)
        )

        if is_heading:
            if current_paras:
                section_text = " ".join(current_paras).strip()
                if len(section_text) >= 20:
                    sections.append({
                        "heading": current_heading,
                        "level": current_level,
                        "text": section_text,
                    })
                current_paras = []
            current_heading = text
            current_level = hlevel
        else:
            current_paras.append(text)

    # Flush remaining
    if current_paras:
        section_text = " ".join(current_paras).strip()
        if len(section_text) >= 20:
            sections.append({
                "heading": current_heading,
                "level": current_level,
                "text": section_text,
            })

    if not sections:
        sections.append({
            "heading": pdf_path.stem,
            "level": 1,
            "text": total_clean_text,
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
                # Strip boilerplate lines from HTML too
                if is_boilerplate_line(text):
                    continue
                if FAIR_WORK_DISCLAIMER_RE.search(text):
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

    # Check for garbling in HTML too
    all_text = " ".join(s["text"] for s in sections)
    ratio = il_ratio(all_text)
    if ratio < MIN_IL_RATIO:
        raise ValueError(
            f"FATAL: Garbling detected for [{source_id}] ({html_path.name}). "
            f"i/l word ratio = {ratio:.3f} (minimum {MIN_IL_RATIO})."
        )

    return merge_short_sections(sections, min_words)


def validate_sources_csv(source_file: Path) -> None:
    """Refuse to run if sources.csv has fewer rows than sources_draft.csv.

    The stale worked-example rows in sources.csv are a trap. Until the human team
    promotes the draft, extraction should use --draft or this check will block.
    """
    if source_file == SOURCES_PATH and DRAFT_SOURCES_PATH.exists():
        with open(SOURCES_PATH, "r", encoding="utf-8") as f:
            sources_count = sum(1 for row in csv.DictReader(f) if row.get("source_id", "").strip())
        with open(DRAFT_SOURCES_PATH, "r", encoding="utf-8") as f:
            draft_count = sum(1 for row in csv.DictReader(f) if row.get("source_id", "").strip())

        if sources_count < draft_count:
            raise RuntimeError(
                f"BLOCKED: data/sources.csv has {sources_count} source(s) but "
                f"data/sources_draft.csv has {draft_count}. "
                f"The production file still holds stale worked-example rows. "
                f"Either promote the draft (copy sources_draft.csv → sources.csv after "
                f"filling in dates and licences) or run with --draft flag."
            )


def run_extraction(use_draft: bool = False) -> None:
    source_file = DRAFT_SOURCES_PATH if use_draft or not SOURCES_PATH.exists() else SOURCES_PATH

    if not source_file.exists():
        print(f"Error: {source_file} not found.", file=sys.stderr)
        sys.exit(1)

    # Safety check: block stale sources.csv
    validate_sources_csv(source_file)

    if fitz is None:
        print(
            "WARNING: PyMuPDF (fitz) not installed. Falling back to pypdf for PDFs. "
            "Install PyMuPDF for better extraction: pip install PyMuPDF",
            file=sys.stderr,
        )

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
                if fitz is not None:
                    sections = extract_pdf_sections_fitz(raw_path, source_id, min_words)
                else:
                    # Fallback to old pypdf (not recommended)
                    from pypdf import PdfReader
                    sections = _extract_pdf_sections_pypdf_fallback(raw_path, source_id, min_words)
            elif raw_path.suffix.lower() in [".html", ".htm"]:
                sections = extract_html_sections(raw_path, source_id, min_words)
            else:
                raise ValueError(f"Unsupported file format: {raw_path.suffix}")

            # Write intermediate JSON
            interim_json = INTERIM_DIR / f"{source_id}.json"
            with open(interim_json, "w", encoding="utf-8") as out:
                json.dump(sections, out, indent=2, ensure_ascii=False)

            # Compute quality metrics
            all_text = " ".join(s["text"] for s in sections)
            ratio = il_ratio(all_text)
            word_counts = [len(s["text"].split()) for s in sections]
            headings = [s["heading"] for s in sections]

            # Write candidate metadata
            meta_json = INTERIM_DIR / f"{source_id}_meta.json"
            meta_data = {
                "source_id": source_id,
                "raw_file": raw_path.name,
                "section_count": len(sections),
                "total_characters": sum(len(s["text"]) for s in sections),
                "il_ratio": round(ratio, 3),
                "min_words_per_section": min(word_counts) if word_counts else 0,
                "max_words_per_section": max(word_counts) if word_counts else 0,
                "headings": headings,
                "url": row.get("url", ""),
                "publisher": row.get("publisher", ""),
                "title": row.get("title", ""),
                "detected_date": row.get("last_updated", ""),
            }
            with open(meta_json, "w", encoding="utf-8") as out:
                json.dump(meta_data, out, indent=2, ensure_ascii=False)

            success_count += 1
            total_sections_count += len(sections)
            print(
                f"  [OK] {source_id}: {len(sections)} sections, "
                f"{meta_data['total_characters']} chars, "
                f"i/l={ratio:.3f}, "
                f"words=[{min(word_counts)}-{max(word_counts)}] "
                f"({raw_path.name})"
            )

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

    # Summary report
    print("\n" + "=" * 60)
    print("PER-SOURCE EXTRACTION REPORT")
    print("=" * 60)
    for row in sources:
        source_id = row.get("source_id", "")
        if not source_id:
            continue
        meta_path = INTERIM_DIR / f"{source_id}_meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            print(f"\n  {source_id}: {meta.get('title', 'N/A')}")
            print(f"    Passages: {meta['section_count']}")
            print(f"    Headings: {meta.get('headings', [])}")
            print(f"    Words per passage: {meta.get('min_words_per_section', '?')} – {meta.get('max_words_per_section', '?')}")
            print(f"    i/l ratio: {meta.get('il_ratio', '?')}")
        else:
            print(f"\n  {source_id}: FAILED (no output)")


def _extract_pdf_sections_pypdf_fallback(pdf_path: Path, source_id: str, min_words: int) -> list[dict]:
    """Legacy pypdf fallback — only used if PyMuPDF is not installed."""
    from pypdf import PdfReader

    reader = PdfReader(str(pdf_path))
    num_pages = len(reader.pages)
    if num_pages == 0:
        raise ValueError(f"[{source_id}] PDF {pdf_path.name} has 0 pages.")

    full_text_chunks = []
    for page_idx, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        if page_text.strip():
            full_text_chunks.append(page_text)

    all_text = "\n".join(full_text_chunks)
    lines = all_text.split("\n")
    cleaned = strip_boilerplate_lines(lines)
    clean_text = "\n".join(cleaned).strip()

    if len(clean_text) < MIN_EXTRACTED_CHARS:
        raise ValueError(
            f"FATAL: Extraction failed for [{source_id}] ({pdf_path.name}). "
            f"After boilerplate stripping, only {len(clean_text)} characters remain."
        )

    ratio = il_ratio(clean_text)
    if ratio < MIN_IL_RATIO:
        raise ValueError(
            f"FATAL: Garbling detected for [{source_id}] ({pdf_path.name}). "
            f"i/l word ratio = {ratio:.3f} (minimum {MIN_IL_RATIO})."
        )

    # Simple single-section fallback
    sections = [{
        "heading": pdf_path.stem,
        "level": 1,
        "text": clean_text,
    }]
    return merge_short_sections(sections, min_words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract text from registered raw documents")
    parser.add_argument("--draft", action="store_true", help="Extract using data/sources_draft.csv")
    args = parser.parse_args()
    run_extraction(use_draft=args.draft)
