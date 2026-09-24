from pathlib import Path
from pypdf import PdfReader

files = {
    '2026 HTML snapshot': Path('data/raw/home_affairs_visa_conditions_20260827_current.html'),
    '2023 HTML snapshot': Path('data/raw/home_affairs_visa_conditions_20230605_superseded.html'),
    'conditions-list (PDF2)': Path('data/raw/Check visa details and conditions2.pdf'),
    'original see-your-visa-conditions (PDF1 in excluded)': Path('data/raw_excluded/Check visa details and conditions.pdf')
}

terms = ['8105', 'fortnight', '48 hours', '40 hours', 'work limitation']

print('=' * 80)
print('EXACT TERM PRESENCE REPORT:')
print('=' * 80)

for name, path in files.items():
    if not path.exists():
        print(f'{name}: FILE NOT FOUND at {path}')
        continue
    if path.suffix == '.html':
        text = path.read_text(encoding='utf-8', errors='ignore')
    else:
        r = PdfReader(str(path))
        text = ' '.join(p.extract_text() or '' for p in r.pages)
    
    print(f'\n{name} ({path.name}): length = {len(text)} characters')
    for term in terms:
        count = text.lower().count(term.lower())
        print(f'  - "{term}": {count} occurrences')
