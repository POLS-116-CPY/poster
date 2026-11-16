"""Assemble the 全開 portrait print edition from the web poster.

python scripts/build_print.py
Requires segno (pip install segno) for the QR code; generated locally, no online service.

The print edition must be a subset of index.html (issue #29): figures and the ΔIC
table are copied verbatim, and every number in the shortened prose must already
appear in the web text, so a typo cannot introduce a value the evidence check
(verify_evidence.py) never saw.
"""
from pathlib import Path
import html
import io
import re
import sys

import segno

root = Path(__file__).resolve().parents[1]
web = (root / 'index.html').read_text(encoding='utf-8')
template = (root / 'print' / 'zenkai.template.html').read_text(encoding='utf-8')
WEB_URL = 'https://pols-116-cpy.github.io/poster/'


def block(tag, marker):
    # Figures without an id are matched by a distinctive attribute fragment instead.
    m = re.search(rf'<{tag}\b[^>]*{re.escape(marker)}[^>]*>.*?</{tag}>', web, re.S)
    if not m:
        sys.exit(f'{tag} with {marker!r} not found in index.html')
    return m.group(0)


def qr():
    buf = io.BytesIO()
    segno.make(WEB_URL, error='m').save(buf, kind='svg', border=0, xmldecl=False,
                                        nl=False, omitsize=True, dark='#1B2A4A',
                                        svgclass='qr')
    return buf.getvalue().decode()


def inject(m):
    kind, key = m.group(1), m.group(2)
    if kind == 'qr':
        return qr()
    attr = key if '=' in key else f'id="{key}"'
    return block(kind, attr)


out = re.sub(r'<!--@(svg|table|qr):?([^>]*?)-->', inject, template)


def visible(doc):
    doc = re.sub(r'<(style|script|title|desc)\b.*?</\1>', ' ', doc, flags=re.S)
    doc = re.sub(r'<svg class="qr".*?</svg>', ' ', doc, flags=re.S)
    doc = re.sub(r'<[^>]+>', ' ', doc)
    return html.unescape(doc).replace('−', '-').replace('–', '-')


NUM = r'[+-]?\d[\d,]*(?:\.\d+)?%?'
print_text = visible(out)

# Numbers: each whole token, with its sign and decimals, must occur in the web text.
# Whole tokens, not substrings: a mistyped 0.07 would otherwise pass inside 0.074.
web_tokens = set(re.findall(NUM, visible(web)))
tokens = sorted(set(re.findall(NUM, print_text)))
missing = [t for t in tokens if t not in web_tokens]
if missing:
    sys.exit(f'numbers absent from index.html: {missing}')

# References: print has no list (the QR leads to the web edition's), so every work cited
# in print must be an entry of the web reference list.
cites = set()
for group in re.findall(r'[（(]([^（()）]*\d{4}[^（()）]*)[）)]', print_text) + \
        re.findall(r'([A-Z][^，。；（()]*?\s*[（(]\d{4}[）)])', print_text):
    for part in re.split(r'[;；、]', group):
        m = re.search(r'([A-Z][A-Za-z–\- ]+?)(?:,| et al\.| &| and|\s*[（(])', part)
        y = re.search(r'(\d{4})', part)
        if m and y:
            cites.add((m.group(1).strip(), y.group(1)))
listed = set()
for entry in re.findall(r'<li>(.*?)</li>', web.partition('class="references"')[2].partition('</ul>')[0], re.S):
    t = visible(entry)
    listed.add((re.match(r'\s*([^,]+),', t).group(1).strip(), re.search(r'\((\d{4})\)', t).group(1)))
if not cites <= listed:
    sys.exit(f'cited in print but not in the web reference list: {sorted(cites - listed)}')

(root / 'print' / 'zenkai.html').write_text(out, encoding='utf-8')
han = len(re.findall(r'[一-鿿]', print_text))
print(f'print/zenkai.html: {len(tokens)} numeric tokens found in index.html; '
      f'{len(cites)} citations in the web reference list; {han} Han characters')
