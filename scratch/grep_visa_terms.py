from pathlib import Path
from pypdf import PdfReader
from bs4 import BeautifulSoup

terms = ['8105', 'fortnight', 'work condition', 'student visa']

print('=' * 80)
print('GREP SEARCH FOR VISA TERMS ACROSS DATA/RAW:')
print('=' * 80)

for p in sorted(Path('data/raw').glob('*')):
    if p.name.startswith('.'):
        continue
    text = ''
    if p.suffix.lower() == '.pdf':
        try:
            r = PdfReader(str(p))
            text = ' '.join(page.extract_text() or '' for page in r.pages)
        except Exception:
            continue
    elif p.suffix.lower() in ['.html', '.htm']:
        try:
            soup = BeautifulSoup(p.read_text(encoding='utf-8', errors='ignore'), 'html.parser')
            text = soup.get_text()
        except Exception:
            continue
            
    matches = {t: text.lower().count(t.lower()) for t in terms}
    if any(matches.values()):
        print(f'{p.name} ({len(text)} chars):')
        for t, count in matches.items():
            if count > 0:
                print(f'   "{t}": {count}')
