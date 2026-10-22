"""Draw the ΔIC equivalence figure from the research repo's aggregate export.

python scripts/build_equivalence_chart.py --research-root /path/to/POLS-116-CPY
Reads docs/findings/24-delta-ic-equivalence.csv (24 cells, post hoc bounds) and
rewrites the block between the equivalence-chart markers in index.html.
"""
from pathlib import Path
import argparse
import csv
import re

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--research-root', type=Path, required=True)
args = parser.parse_args()
rows = list(csv.DictReader((args.research_root / 'docs/findings/24-delta-ic-equivalence.csv').read_text(encoding='utf-8-sig').splitlines()))
assert len(rows) == 24, len(rows)

# Shared with scripts/verify_evidence.py, which re-derives every coordinate from these constants.
X0, X1, LO, HI = 190, 645, -0.055, 0.03
Y0, ROW, GAP = 56, 16, 8
H = Y0 + 8 * 3 * ROW + 7 * GAP + 48
industries = [('半導體業', '半導體'), ('金融保險業', '金融')]
horizons = [5, 20, 30, 60]
baselines = [('market', '市場'), ('sector', '產業'), ('allocation', '配置')]


def sx(v):
    return X0 + (float(v) - LO) / (HI - LO) * (X1 - X0)


def f(v):
    return f'{v:.3f}'.rstrip('0').rstrip('.')


out = ['<div aria-label="ΔIC 等效界限圖" class="chart-scroll" role="region" tabindex="0">',
       f'<svg aria-labelledby="eq-title eq-desc" id="equivalence-chart" role="img" viewbox="0 0 660 {H}">',
       '<title id="eq-title">Ridge 雙軌 ΔIC 的單尾上界與事後等效界限（24 格）</title>',
       '<desc id="eq-desc">每列為一個基準、產業與期距。圓點為 ΔIC 點估計，實線延伸到未校正單尾 95% 上界，虛線再延伸到 Bonferroni-8 上界；紅色短豎線為該期距的基本定律界限 δ_H，點線為固定 0.01 界限。上界落在界限左側即排除該幅度的改善。</desc>']
# Legend: two lines above the plot.
out += [f'<circle cx="14" cy="12" fill="var(--accent)" r="3.5"></circle><text fill="var(--ink-2)" font-size="12" x="24" y="16">ΔIC 點估計</text>',
        f'<line stroke="var(--accent)" stroke-width="2" x1="108" x2="136" y1="12" y2="12"></line><text fill="var(--ink-2)" font-size="12" x="142" y="16">未校正單尾 95% 上界</text>',
        f'<line stroke="var(--accent)" stroke-dasharray="4 3" stroke-width="1.5" x1="282" x2="310" y1="12" y2="12"></line><text fill="var(--ink-2)" font-size="12" x="316" y="16">Bonferroni-8 上界</text>',
        f'<line stroke="var(--neg)" stroke-width="2.5" x1="14" x2="14" y1="26" y2="40"></line><text fill="var(--ink-2)" font-size="12" x="24" y="38">δ<tspan baseline-shift="sub" font-size="9">H</tspan> 基本定律界限</text>',
        f'<line stroke="var(--ink-3)" stroke-dasharray="1.5 3" stroke-width="1.5" x1="142" x2="142" y1="26" y2="40"></line><text fill="var(--ink-2)" font-size="12" x="152" y="38">固定 0.01 敏感度界限</text>']
bottom = Y0 + 8 * 3 * ROW + 7 * GAP
for k in range(-5, 4):
    v = k / 100
    out.append(f'<line stroke="var(--grid)" x1="{f(sx(v))}" x2="{f(sx(v))}" y1="{Y0 - 4}" y2="{bottom + 4}"></line>')
    label = '0' if k == 0 else f'{v:+.2f}'.replace('-', '−')
    out.append(f'<text fill="var(--ink-2)" font-size="11" text-anchor="middle" x="{f(sx(v))}" y="{bottom + 20}">{label}</text>')
out.append(f'<line stroke="var(--ink-3)" stroke-width="1.2" x1="{f(sx(0))}" x2="{f(sx(0))}" y1="{Y0 - 4}" y2="{bottom + 4}"></line>')
out.append(f'<line data-bound="fixed" data-value="0.01" stroke="var(--ink-3)" stroke-dasharray="1.5 3" stroke-width="1.5" x1="{f(sx(.01))}" x2="{f(sx(.01))}" y1="{Y0 - 4}" y2="{bottom + 4}"></line>')
out.append(f'<text fill="var(--ink-2)" font-size="12" text-anchor="middle" x="{f((X0 + X1) / 2)}" y="{bottom + 40}">ΔIC = IC(M1) − IC(M0)；界限以右為「未能排除」</text>')
for g, ((ind, ind_label), h) in enumerate((i, h) for i in industries for h in horizons):
    top = Y0 + g * (3 * ROW + GAP)
    cells = [next(r for r in rows if r['source'] == b and r['industry'] == ind and int(r['horizon']) == h) for b, _ in baselines]
    delta_fl = float(cells[0]['delta_fl'])
    assert all(abs(float(c['delta_fl']) - delta_fl) < 1e-15 for c in cells)
    out.append(f'<text fill="var(--ink)" font-size="12" x="8" y="{top + ROW + 10.5}">{ind_label} {h} 日</text>')
    out.append(f'<line data-bound="fl" data-horizon="{h}" data-industry="{ind}" data-value="{delta_fl!r}" stroke="var(--neg)" stroke-width="2.5" '
               f'x1="{f(sx(delta_fl))}" x2="{f(sx(delta_fl))}" y1="{top - 1}" y2="{top + 3 * ROW + 1}"></line>')
    for r, ((b, b_label), c) in enumerate(zip(baselines, cells)):
        y = top + r * ROW + ROW / 2
        est, ub, ub8 = float(c['delta_ic']), float(c['ub_unadj']), float(c['ub_bonf8'])
        attrs = f'data-baseline="{b}" data-horizon="{h}" data-industry="{ind}"'
        out.append(f'<text fill="var(--ink-2)" font-size="11" text-anchor="end" x="176" y="{f(y + 4)}">{b_label}</text>')
        out.append(f'<line {attrs} data-ub-bonf8="{ub8!r}" stroke="var(--accent)" stroke-dasharray="4 3" stroke-width="1.5" x1="{f(sx(ub))}" x2="{f(sx(ub8))}" y1="{f(y)}" y2="{f(y)}"></line>')
        out.append(f'<line {attrs} data-ub-unadj="{ub!r}" stroke="var(--accent)" stroke-width="2" x1="{f(sx(est))}" x2="{f(sx(ub))}" y1="{f(y)}" y2="{f(y)}"></line>')
        out.append(f'<circle {attrs} cx="{f(sx(est))}" cy="{f(y)}" data-value="{est!r}" fill="var(--accent)" r="3.5"></circle>')
out.append('</svg></div>')
out.append('<div class="note">每列為一個基準 × 產業 × 期距格，市場、產業、配置依序排列。Bonferroni-8 下唯一越過基本定律界限的是配置基準半導體 H=60（上界 0.0254 對 0.0223），屬事前僅作描述的期距；越過固定 0.01 界限的有六格，其中市場基準四格（金融保險 H=20／30／60、半導體 H=60）。校正後，市場基準金融保險業長期距只能排除約 0.013–0.022 以上的改善。標準誤為匯出的 Newey–West（落後期 H−1）。</div>')

index = Path(__file__).resolve().parents[1] / 'index.html'
html = index.read_text(encoding='utf-8')
begin, end = '<!-- BEGIN equivalence-chart -->', '<!-- END equivalence-chart -->'
assert html.count(begin) == 1 and html.count(end) == 1
html = re.sub(re.escape(begin) + '.*?' + re.escape(end), lambda _: begin + '\n' + '\n'.join(out) + '\n' + end, html, flags=re.S)
index.write_text(html, encoding='utf-8')
print(f'wrote {len(out)} chart lines')
