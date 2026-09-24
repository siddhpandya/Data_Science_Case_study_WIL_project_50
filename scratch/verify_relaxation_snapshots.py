import requests
import hashlib
from bs4 import BeautifulSoup
from pathlib import Path
import re

snapshots = {
    "historical_20220607": "http://web.archive.org/web/20220607122248/https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/student-500/temporary-relaxation-of-working-hours-for-student-visa-holders",
    "post_relaxation_20230831": "http://web.archive.org/web/20230831101024/https://immi.homeaffairs.gov.au/visas/getting-a-visa/visa-listing/student-500/temporary-relaxation-of-working-hours-for-student-visa-holders"
}

headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
out_dir = Path("scratch/test_snapshots")
out_dir.mkdir(parents=True, exist_ok=True)

terms_to_check = ["student visa", "hours", "fortnight", "work", "relaxation", "course"]

print("=" * 80)
print("VERIFYING CANDIDATE STATIC RELAXATION PAGE SNAPSHOTS:")
print("=" * 80)

for label, url in snapshots.items():
    print(f"\nFetching {label} from:\n  {url}")
    resp = requests.get(url, headers=headers, timeout=30)
    if resp.status_code != 200:
        print(f"FAILED to fetch {label}: status {resp.status_code}")
        continue
        
    raw_bytes = resp.content
    sha256 = hashlib.sha256(raw_bytes).hexdigest()
    out_file = out_dir / f"{label}.html"
    out_file.write_bytes(raw_bytes)
    print(f"  -> Saved {out_file.name}: {len(raw_bytes)} bytes, sha256: {sha256}")
    
    # Parse with BeautifulSoup
    soup = BeautifulSoup(raw_bytes.decode("utf-8", errors="ignore"), "html.parser")
    
    # Strip Wayback archive toolbar and chrome
    for el in soup.find_all(id=re.compile(r"wm-ipp|wm-capinfo|wm-share|donato", re.I)):
        el.decompose()
    for el in soup.find_all(class_=re.compile(r"wm-|wb-autocomplete", re.I)):
        el.decompose()
        
    # Strip navigation, header, footer, scripts, styles, breadcrumbs, search
    for tag in soup(["script", "style", "noscript", "svg", "nav", "header", "footer", "aside"]):
        tag.decompose()
    for el in soup.find_all(class_=re.compile(r"breadcrumb|search|navigation|nav-container|page-header|site-header|footer", re.I)):
        el.decompose()
    for el in soup.find_all(id=re.compile(r"breadcrumb|search|navigation|nav-container|header|footer", re.I)):
        el.decompose()
        
    # Locate main body content
    # On Home Affairs pages, content is typically in div.content-wrapper, div#content, or main
    main = (
        soup.find("main") or
        soup.find("article") or
        soup.find(class_=re.compile(r"landing-content|body-content|page-content|main-content", re.I)) or
        soup.find(id=re.compile(r"content|main", re.I)) or
        soup.body
    )
    
    # Extract clean text paragraphs
    paragraphs = []
    for elem in main.find_all(["h1", "h2", "h3", "h4", "p", "li"]):
        txt = elem.get_text().strip()
        # skip short navigation elements
        if txt and len(txt) > 20 and not any(nav_word in txt.lower() for nav_word in ["skip to", "popular searches", "your previous searches"]):
            paragraphs.append(txt)
            
    clean_text = "\n\n".join(paragraphs)
    char_count = len(clean_text)
    word_count = len(clean_text.split())
    
    print(f"  -> Clean extracted length: {char_count} characters, {word_count} words")
    print(f"  -> Exceeds 1,000 characters: {char_count > 1000}")
    
    print("  -> Term occurrences:")
    for t in terms_to_check:
        c = clean_text.lower().count(t.lower())
        print(f"       \"{t}\": {c} occurrences")
        
    print(f"  -> First 500 characters of clean prose:")
    print("-" * 50)
    print(clean_text[:500].encode("ascii", errors="replace").decode("ascii"))
    print("-" * 50)
