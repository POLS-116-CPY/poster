"""Verify the public poster against aggregate exports, never raw TEJ data.

python scripts/verify_evidence.py --research-root /path/to/POLS-116-CPY [--out review/issue-28]
Requires beautifulsoup4. Writes the reviewed values and LF-normalized hashes.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from decimal import Decimal, ROUND_HALF_UP
import subprocess
from bs4 import BeautifulSoup

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--research-root', type=Path, required=True)
parser.add_argument('--out', default='review/issue-28', help='review folder that receives evidence.json')
args = parser.parse_args()
root = args.research_root.resolve()
poster = Path(__file__).resolve().parents[1]
soup = BeautifulSoup((poster / 'index.html').read_text(encoding='utf-8'), 'html.parser')
sources = {}
checks = []


def read(path):
    assert path.startswith(('results/', 'docs/findings/', 'docs/paper/')), path
    text = (root / path).read_text(encoding='utf-8-sig')
    sources[path] = hashlib.sha256(text.replace('\r\n', '\n').encode()).hexdigest()
    return text


def csvrows(path):
    return list(csv.DictReader(read(path).splitlines()))


def js(path):
    return json.loads(read(path))


def norm(s):
    return re.sub(r'\s+', '', s.replace('−', '-').replace('–', '-'))


def eq(label, actual, expected):
    assert norm(str(actual)) == norm(str(expected)), f'{label}: {actual!r} != {expected!r}'
    checks.append({'label': label, 'value': expected})


def near(label, actual, expected, tolerance=1e-6):
    assert math.isclose(float(actual), float(expected), abs_tol=tolerance), f'{label}: {actual} != {expected}'
    checks.append({'label': label, 'value': expected})


def cells(id):
    return [[c.get_text(' ', strip=True) for c in tr.find_all(['td', 'th'], recursive=False)]
            for tr in soup.select(f'#{id} tbody tr')]


text_of = lambda sel: ' '.join(e.get_text(' ', strip=True) for e in soup.select(sel))


def signed(value, decimals=2):
    return f'{float(value):+.{decimals}f}'


industries = ['半導體業', '金融保險業']
horizons = [5, 20, 30, 60]
inference = js('docs/findings/18-delta-ic-inference.json')
evidence = js('docs/findings/17-three-baseline-evidence.json')
delta = cells('delta-table')
for col, baseline in enumerate(['market', 'sector', 'allocation'], 1):
    data = [r for r in inference['delta_ic'] if r['source'] == baseline and r['candidate'] == 'M1']
    for row, (industry, h) in enumerate((i, h) for i in industries for h in horizons):
        r = next(r for r in data if r['industry'] == industry and r['horizon'] == h)
        eq(f'ΔIC/{baseline}/{industry}/{h}', delta[row][col], signed(r['delta_ic'], 4))
    eq(f'NW min/{baseline}', delta[8][col], f'{min(r["p_one_sided_gt0"] for r in data):.3f}')
    ranks = [r for r in inference['perm_rank'] if r['baseline'] == baseline]
    eq(f'ΔIC permutation min/{baseline}', delta[9][col], f'{min(r["p_rank"] for r in ranks):.2f}')
    if baseline == 'market':
        a = [r for r in inference['delta_ic'] if r['source'] == 'market' and r['candidate'] == 'M1_noPrice' and r['industry'] == industries[1]]
        assert all(r['delta_ic'] > 0 and r['p_one_sided_gt0'] > .05 for r in a)
        eq('track A financial lower ΔIC', f'{min(r["delta_ic"] for r in a):+.4f}', '+0.0016')
        eq('track A financial upper ΔIC', f'{max(r["delta_ic"] for r in a):+.4f}', '+0.0039')
        eq('track A financial lower p', f'{min(r["p_one_sided_gt0"] for r in a):.2f}', '0.26')
        eq('track A financial upper p', f'{max(r["p_one_sided_gt0"] for r in a):.2f}', '0.43')

decomp = csvrows('results/16_decompose/decomposition.csv')
ridge_d = [r for r in decomp if r['model'] == 'ridge']
assert len(ridge_d) == 8 and all(float(r['r2_pooled']) < 0 and float(r['r2_cross_sectional']) < 0 for r in ridge_d)
assert re.search(r'Ridge 的八格合併 R² 與選股 R²[^。]*都是負的', soup.get_text())
checks.append({'label': 'Ridge pooled and cross-sectional R² all negative', 'value': '8/8'})
xgb_d = [r for r in decomp if r['model'] == 'xgboost' and float(r['r2_pooled']) > 0]
eq('XGBoost pooled-positive cells', len(xgb_d), 6)
eq('of which cross-sectional negative', sum(float(r['r2_cross_sectional']) < 0 for r in xgb_d), 5)
assert '六格中，五格的選股 R² 為負' in soup.get_text()
# The ΔIC table's meaning line: M0 IC levels and the share of ranking ability lost (market baseline, Ridge).
base = [r for r in csvrows('results/sweep/robustness_table.csv') if r['variant'] == 'base' and r['model'] == 'ridge']
for ind, lvl, loss in [('半導體業', '0.04–0.08', '26%–74%'), ('金融保險業', '0.07–0.15', '8%–14%')]:
    rows = [r for r in base if r['industry'] == ind]
    assert len(rows) == 4
    ic0 = [float(r['mean_ic_M0']) for r in rows]
    rel = [1 - float(r['mean_ic_M1']) / float(r['mean_ic_M0']) for r in rows]
    for r in rows:
        d = next(x for x in inference['delta_ic'] if x['source'] == 'market' and x['candidate'] == 'M1' and x['industry'] == ind and x['horizon'] == int(r['horizon']))
        near(f'IC difference matches ΔIC/{ind}/{r["horizon"]}', float(r['mean_ic_M1']) - float(r['mean_ic_M0']), d['delta_ic'], 1e-9)
    eq(f'M0 IC range/{ind}', f'{min(ic0):.2f}–{max(ic0):.2f}', lvl)
    eq(f'ranking ability lost/{ind}', f'{min(rel):.0%}–{max(rel):.0%}', loss)
    assert lvl in text_of('#results') and loss in text_of('#results')

for bar in soup.select('#decomposition-chart rect[data-value]'):
    r = next(r for r in decomp if r['model'] == 'xgboost' and r['industry'] == bar['data-industry'] and r['horizon'] == bar['data-horizon'])
    v = float(r[bar['data-metric']])
    near('XGBoost bar data', bar['data-value'], v)
    near('XGBoost bar origin', bar['x'], 330 + min(0, v*100)*95)
    near('XGBoost bar width', bar['width'], abs(v)*100*95)
    eq('XGBoost visible label', bar.find_next_sibling('text').get_text(), signed(v*100)+'%')

mde = [r for r in csvrows('results/15_power/mde.csv') if r['threshold'] == 'BH_most_conservative' and r['sample'] in industries]
mde_h = lambda h: [float(r['mde_pct']) for r in mde if int(r['horizon']) == h]
# Order of magnitude only: H=5 exports are 0.05–0.06%, H=60 exports are 0.45–0.50%.
assert {round(v, 2) for v in mde_h(5)} <= {.05, .06} and {round(v, 2) for v in mde_h(60)} <= {.45, .5}
checks.append({'label': 'MDE order of magnitude', 'value': '0.05% (H=5) to 0.5% (H=60)'})
spread = csvrows('docs/findings/25-mde-spread.csv')
fin20 = next(r for r in spread if r['sample'] == '金融保險業' and r['threshold'] == 'BH_most_conservative' and r['horizon'] == '20')
exported = next(float(r['mde_pct']) for r in mde if r['sample'] == '金融保險業' and r['horizon'] == '20')
eq('MDE seeds', fin20['count'], '21')
eq('MDE financial H20 exported is the seed maximum', f'{exported:.2f}', f'{float(fin20["max"]):.2f}')
eq('MDE financial H20 exported', f'{exported:.2f}%', '0.25%')
eq('MDE financial H20 median', f'{float(fin20["median"]):.2f}%', '0.17%')

placebo = csvrows('results/sweep/placebo_ranks.csv')
heat = [c.get_text(strip=True) for c in soup.select('.hm .cl')]
for row, (industry, model) in enumerate((i, m) for i in industries for m in ['ridge', 'xgboost']):
    for col, h in enumerate(horizons):
        r = next(r for r in placebo if r['placebo_kind'] == 'permute' and r['industry'] == industry and r['model'] == model and int(r['horizon']) == h)
        eq(f'R² permutation/{industry}/{model}/{h}', heat[row*4+col], f'{float(r["p_rank_r2"]):.2f}')
for bar in soup.select('#cw-chart rect[data-value]'):
    r = next(r for r in placebo if r['placebo_kind'] == 'permute' and all(r[k] == bar['data-'+k] for k in ['industry', 'model', 'horizon']))
    v = float(r['reject_rate'])
    near('CW bar data', bar['data-value'], v)
    near('CW bar y', bar['y'], 205-v*155)
    near('CW bar height', bar['height'], v*155)
    eq('CW visible label', bar.find_next_sibling('text').get_text(), f'{v:.0%}')

support = csvrows('results/11_matching/common_support.csv')
for point in soup.select('#matching-chart circle[data-value]'):
    v = statistics.median(float(r['pct_fin_matchable']) for r in support if r['quarter'].startswith(point['data-year']))
    near('annual matchable fraction', point['data-value'], v)
    near('annual matchable position', point['cy'], 145-v*110)
    eq('annual matchable label', point.find_next_sibling('text').get_text(), f'{v:.1%}')
census = csvrows('results/11_matching/quarter_census.csv')
kept = [r for r in census if r['quarter'] >= '2018Q1' and r['kept'] == 'True']
eq('kept evaluation quarters', len(kept), 9)
eq('last kept quarter', max(r['quarter'] for r in kept), '2021Q1')

gat = [float(r['r2_oos']) for r in csvrows('results/13_gat/summary.csv') if r['baseline'] == 'M0']
assert len(gat) == 8 and max(gat) < 0
eq('GAT R² upper', f'{max(gat)*100:.1f}%', '-2.3%')
eq('GAT R² lower', f'{min(gat)*100:.1f}%', '-11.9%')
eq('GAT R² mean', f'{statistics.mean(gat)*100:.1f}%', '-7.9%')
gat_ic = csvrows('docs/findings/25-gat-delta-ic.csv')
vs0 = [r for r in gat_ic if r['comparison'] == 'GAT_M1-ridge_M0']
vs1 = [r for r in gat_ic if r['comparison'] == 'GAT_M1-ridge_M1']
eq('GAT ΔIC vs Ridge M0 negative', sum(float(r['delta_ic']) < 0 for r in vs0), 8)
eq('GAT ΔIC vs Ridge M1 positive', f'{sum(float(r["delta_ic"]) > 0 for r in vs1)}/{len(vs1)}', '3/8')
eq('GAT ΔIC min p', f'{min(float(r["p_one_sided_gt0"]) for r in gat_ic):.3f}', '0.435')

fm = csvrows('results/18_fama_macbeth/fama_macbeth.csv')
fm_p = [float(r['nw_p']) for r in fm if r['model'] == 'ridge' and r['variant'] == 'M1_incremental']
assert len(fm_p) == 8
eq('FM p range', f'{min(fm_p):.2f}–{max(fm_p):.2f}', '0.26–0.90')
assert 'p=0.26–0.90' in text_of('#economic')

for row, baseline in enumerate(['market', 'sector', 'allocation']):
    b = evidence['baselines'][baseline]
    groups = [b['total']] + [b['groups'][g] for g in ['substantive_including_base', 'permute', 'noise']]
    for col, group in enumerate(groups, 1):
        eq(f'robustness/{baseline}/{col}', cells('robustness-table')[row][col], f'{group["positive_cells"]}/{group["cells"]}')
    for col, model in enumerate(['ridge', 'xgboost'], 5):
        rates = [r['reject_rate'] for r in b['placebo'] if r['model'] == model]
        eq(f'CW range/{baseline}/{model}', cells('robustness-table')[row][col], f'{min(rates):.3f}–{max(rates):.3f}')

costs = csvrows('results/14_economic/cost_matrix.csv')
for col, spec in enumerate(['M1', 'M0'], 1):
    conservative = [r for r in costs if r['spec'] == spec and r['horizon'] == '20' and r['cost_scenario'] == 'retail' and float(r['sbl']) == .03 and float(r['lambda_mi']) == .003]
    eq(f'H20 positive/{spec}', cells('economic-table')[0][col], f'{sum(float(r["net_sharpe"]) > 0 for r in conservative)}/{len(conservative)}')
    all_costs = [r for r in costs if r['spec'] == spec]
    eq(f'full cost grid/{spec}', len(all_costs), 1536)
    eq(f'full cost positive/{spec}', sum(float(r['net_sharpe']) > 0 for r in all_costs), 564 if spec == 'M1' else 957)
    eq(f'full cost max/{spec}', f'{max(float(r["net_sharpe"]) for r in all_costs):.3f}', '0.730' if spec == 'M1' else '0.853')
quarterly = js('docs/findings/20-economic-h60-quarterly.json')
for col, spec in enumerate(['M1', 'M0'], 1):
    r = quarterly['summary']['calendar_quarter'][spec]['most_conservative']
    eq(f'H60 quarterly/{spec}', cells('economic-table')[1][col], f'{r["n_positive"]}/{r["n_cells"]}')

positive = [r for r in decomp if float(r['r2_pooled']) > 0]
eq('positive pooled cells', len(positive), 18)
eq('matching >= 100% cells', sum(float(r['shrink_share']) >= 1 for r in positive), 13)
remainder = [float(r['shrink_share']) for r in positive if float(r['shrink_share']) < 1]
eq('remaining matching lower percent', f'{min(remainder):.0%}', '66%')
eq('remaining matching upper percent', f'{max(remainder):.0%}', '84%')

# Three-baseline teaching example: every displayed number is re-derived from findings/22.
example = js('docs/findings/22-poster-debiasing-example.json')
first_example_check = len(checks)
etfs, ex_stocks = example['meta']['etf_order'], example['stocks']
assert etfs == ['0050', '0053', '0056'] and list(ex_stocks) == ['2330', '2454', '3034', '2379', '2303']


def half_up(value, decimals):
    # 0056's a is exactly 18.15%; float formatting would show 18.1, the decimal value rounds to 18.2.
    return Decimal(repr(round(value, 12))).quantize(Decimal(1).scaleb(-decimals), ROUND_HALF_UP)


def ex_signed(value, decimals=1):
    r = half_up(value, decimals)
    return ('+' if r >= 0 else '−') + f'{abs(r):.{decimals}f}'


def ex_pct(value):
    return f'{half_up(value*100, 1):.1f}%'


def ex_vec(values):
    return '[' + ', '.join(ex_signed(v*100) for v in values) + ']'


weights = {tr['data-code']: [td.get_text(strip=True) for td in tr.find_all('td')] for tr in soup.select('#debias-weights tr[data-code]')}
assert list(weights) == list(ex_stocks), weights.keys()
for code, s in ex_stocks.items():
    eq(f'example name/{code}', weights[code][0], f'{code} {s["name"]}')
    eq(f'example w_mkt/{code}', weights[code][1], ex_pct(s['w_mkt']))
    eq(f'example c/{code}', weights[code][2], ex_pct(s['c_sector']))
    for k, etf in enumerate(etfs):
        eq(f'example B/{code}/{etf}', weights[code][3+k], ex_pct(s['holdings_B'][k]))
a_row = [td.get_text(strip=True) for td in soup.select_one('#debias-weights tr[data-row="a"]').find_all('td')]
n_row = [td.get_text(strip=True) for td in soup.select_one('#debias-weights tr[data-row="n"]').find_all('td')]
for k, etf in enumerate(etfs):
    eq(f'example a/{etf}', a_row[3+k], ex_pct(example['allocation_a_semis'][etf]))
    eq(f'example semis held/{etf}', n_row[3+k], example['n_semis_held_in_U'][etf])
tsmc = ex_stocks['2330']
assert math.isclose(tsmc['a_times_c'][0], example['allocation_a_semis']['0050']*tsmc['c_sector'])
eq('example worked 2330×0050', soup.select_one('#debias-worked').get_text('', strip=True),
   f'實算（台積電 × 0050）：a = {ex_pct(example["allocation_a_semis"]["0050"])}，c = {ex_pct(tsmc["c_sector"])} → '
   f'a·c = {ex_pct(tsmc["a_times_c"][0])}；B = {ex_pct(tsmc["holdings_B"][0])} → B − wmkt = {ex_signed(tsmc["residual_market"][0]*100)}、'
   f'B − c = {ex_signed(tsmc["residual_sector"][0]*100)}、B − a·c = {ex_signed(tsmc["residual_allocation"][0]*100)} 個百分點')

ex_names = {c: s['name'] for c, s in ex_stocks.items()}
ex_cos = {tuple(r['pair']): r for r in example['cosines']}
assert len(ex_cos) == 10
baselines3 = ['market', 'sector', 'allocation']
for b in baselines3:
    lines = soup.select(f'#network-chart line[data-baseline="{b}"]')
    drawn = {tuple(l['data-pair'].split('-')): l for l in lines}
    expected_edges = {p for p, r in ex_cos.items() if abs(r[b]) >= .5}
    eq(f'network edges/{b}', sorted('-'.join(p) for p in drawn), sorted('-'.join(p) for p in expected_edges))
    for p, l in drawn.items():
        v = ex_cos[p][b]
        near(f'edge value/{b}/{"-".join(p)}', l['data-value'], v)
        near(f'edge width/{b}/{"-".join(p)}', l['stroke-width'], 1 + 5*(abs(v) - .5), 1e-4)
        assert (l.get('stroke-dasharray') is not None) == (v < 0) and l['stroke'] == ('var(--accent)' if v > 0 else 'var(--neg)')


def pair_label(i, j, b):
    r = ex_cos.get((i, j)) or ex_cos[(j, i)]
    return f'{ex_names[i]}–{ex_names[j]} {ex_signed(r[b], 3)}'


rest = [r['allocation'] for p, r in ex_cos.items() if '2330' not in p]
tsmc_rest = [r['allocation'] for p, r in ex_cos.items() if '2330' in p and '2454' not in p]
assert min(rest) > 0 and max(tsmc_rest) < 0
expected_notes = [pair_label('3034', '2379', 'market'), pair_label('2454', '2303', 'market'), pair_label('2330', '2454', 'market') + '，無邊',
                  pair_label('3034', '2379', 'sector'), pair_label('2330', '2454', 'sector'), pair_label('2330', '3034', 'sector'),
                  pair_label('2330', '2454', 'allocation'), f'其餘四檔間 {ex_signed(min(rest), 3)}～{ex_signed(max(rest), 3)}',
                  f'台積電–其餘 {ex_signed(min(tsmc_rest), 3)}～{ex_signed(max(tsmc_rest), 3)}']
assert abs(ex_cos[('2330', '2454')]['market']) < .5
for k, (t, e) in enumerate(zip(soup.select('#network-chart text.cos-note'), expected_notes, strict=True)):
    eq(f'network note/{k}', t.get_text(), e)
alloc53 = [s['residual_allocation'][1]*100 for s in ex_stocks.values()]
reading = soup.select_one('#debias-reading').get_text()
for label, value in [('TSMC c', ex_pct(tsmc['c_sector'])), ('TSMC 0050', ex_pct(tsmc['holdings_B'][0])), ('TSMC 0053', ex_pct(tsmc['holdings_B'][1])),
                     ('0053 semis', f'{example["n_semis_held_in_U"]["0053"]} 檔'), ('0053 residual range', f'介於 {ex_signed(min(alloc53))} 與 {ex_signed(max(alloc53))}'),
                     ('0050/0056 semis', f'各只持 {example["n_semis_held_in_U"]["0050"]} 檔'),
                     ('TSMC–MTK market', ex_signed(ex_cos[('2330', '2454')]['market'], 3)), ('TSMC–MTK sector', ex_signed(ex_cos[('2330', '2454')]['sector'], 3)),
                     ('TSMC–MTK allocation', ex_signed(ex_cos[('2330', '2454')]['allocation'], 3))]:
    assert value in reading, (label, value)
    checks.append({'label': f'example reading/{label}', 'value': value})
assert tsmc['holdings_B'][2] == 0 and '0056 未持有' in reading
assert example['n_semis_held_in_U']['0050'] == example['n_semis_held_in_U']['0056']
caption = soup.select_one('#debias-caption').get_text()
for phrase in ['教學示例，非估計結果', '持股取 MOPS（公開資訊觀測站）2023Q4', '亦取 2023Q4', '兩季前公告的持股', '只以圖中三檔 ETF 為維度']:
    assert phrase in caption, phrase
assert 'the three ETFs shown only' == example['meta']['cosine_dimension'] and '2023Q4' in example['meta']['weights'] and '2023Q4' in example['meta']['holdings']
checks.append({'label': 'example caption', 'value': 'holdings and weights 2023Q4; two-quarter pipeline lag; three-ETF cosines; teaching example'})
# Audit: no number in the disclosure may lack a source. Structural tokens are codes, dates, thresholds and section numbers.
rendered = set(re.findall(r'\d+(?:\.\d+)*', ' '.join(str(c['value']) for c in checks[first_example_check:])))
structural = {'0050', '0053', '0056', '2023', '4', '2330', '2454', '3034', '2379', '2303', '0.5', '0.05', '50', '5.1.1', '5.1'}
block = soup.select_one('#network-chart').find_parent('details')
for token in re.findall(r'\d+(?:\.\d+)*', block.get_text(' ')):
    assert token in rendered or token in structural, f'untraced number in three-baseline example: {token}'
checks.append({'label': 'three-baseline disclosure number audit', 'value': 'every numeric token sourced from findings/22 or structural'})

# ---- Issue #28: post hoc evidence from issue #25 ----
paper = read('docs/paper/paper.md')
first_28 = len(checks)
eqv = csvrows('docs/findings/24-delta-ic-equivalence.csv')
eqj = js('docs/findings/24-delta-ic-equivalence.json')
assert len(eqv) == 24
X0, X1, LO, HI = 190, 645, -0.055, 0.03
sx = lambda v: X0 + (float(v) - LO) / (HI - LO) * (X1 - X0)
for r in eqv:
    key = {'data-baseline': r['source'], 'data-industry': r['industry'], 'data-horizon': r['horizon']}
    match = lambda tag: [e for e in soup.select(f'#equivalence-chart {tag}') if all(e.get(k) == v for k, v in key.items())]
    (dot,), lines = match('circle'), match('line')
    near('equivalence estimate', dot['data-value'], r['delta_ic'], 1e-15)
    near('equivalence estimate x', dot['cx'], sx(r['delta_ic']), 1e-3)
    unadj = next(l for l in lines if l.get('data-ub-unadj'))
    bonf = next(l for l in lines if l.get('data-ub-bonf8'))
    near('equivalence unadjusted bound', unadj['data-ub-unadj'], r['ub_unadj'], 1e-15)
    near('equivalence unadjusted x', unadj['x2'], sx(r['ub_unadj']), 1e-3)
    near('equivalence Bonferroni bound', bonf['data-ub-bonf8'], r['ub_bonf8'], 1e-15)
    near('equivalence Bonferroni x', bonf['x2'], sx(r['ub_bonf8']), 1e-3)
for h in horizons:
    for ind in industries:
        (tick,) = [l for l in soup.select('#equivalence-chart line[data-bound="fl"]') if l['data-industry'] == ind and l['data-horizon'] == str(h)]
        near(f'δ_H/{h}', tick['data-value'], eqj['delta_fl'][str(h)], 1e-15)
        near(f'δ_H x/{h}', tick['x1'], sx(eqj['delta_fl'][str(h)]), 1e-3)
near('fixed bound x', soup.select_one('#equivalence-chart line[data-bound="fixed"]')['x1'], sx(eqj['delta_fixed']), 1e-3)
assert eqj['delta_fl_formula'] == '0.25 / sqrt(N*252/H), N=30'
eq('δ_H values', '／'.join(f'{eqj["delta_fl"][str(h)]:.4f}' for h in horizons), '0.0064／0.0129／0.0157／0.0223')
assert '0.0064／0.0129／0.0157／0.0223' in text_of('#equivalence')
excl = lambda k: sum(eqj['cells_excluded'][f'{b}/{k}'] for b in ['market', 'sector', 'allocation'])
eq('unadjusted excluded (FL)', excl('excl_fl_unadj'), 24)
eq('unadjusted excluded (0.01)', excl('excl_fixed_unadj'), 24)
eq('Bonferroni excluded (FL)', f'{excl("excl_fl_bonf8")}/24', '23/24')
eq('Bonferroni excluded (0.01)', f'{excl("excl_fixed_bonf8")}/24', '18/24')
miss_fl = [r for r in eqv if r['excl_fl_bonf8'] == 'False']
eq('only FL miss', [(r['source'], r['industry'], r['horizon']) for r in miss_fl], [('allocation', '半導體業', '60')])
eq('FL miss bound', f'{float(miss_fl[0]["ub_bonf8"]):.4f}', '0.0254')
eq('FL miss threshold', f'{float(miss_fl[0]["delta_fl"]):.4f}', '0.0223')
miss_fixed = [r for r in eqv if r['excl_fixed_bonf8'] == 'False']
eq('0.01 misses', len(miss_fixed), 6)
eq('0.01 market misses', sorted((r['industry'], r['horizon']) for r in miss_fixed if r['source'] == 'market'),
   sorted([('金融保險業', '20'), ('金融保險業', '30'), ('金融保險業', '60'), ('半導體業', '60')]))
fin_long = [float(r['ub_bonf8']) for r in eqv if r['source'] == 'market' and r['industry'] == '金融保險業' and r['horizon'] in ('20', '30', '60')]
eq('market financial long-horizon excludable range', f'{min(fin_long):.3f}–{max(fin_long):.3f}', '0.013–0.022')
assert '0.013–0.022' in paper
eqtext = text_of('#equivalence')
for v in ['24 格', '23/24', '18/24', '0.0254', '0.0223', '0.013–0.022', '0.01']:
    assert v in eqtext, v

power = csvrows('docs/findings/25-delta-ic-power-summary.csv')
pw = lambda ind, h, rho: next(r for r in power if r['industry'] == ind and r['horizon'] == str(h) and float(r['rho']) == rho)
eq('power refits', sum(int(r['draws']) for r in power), 4800)
eq('H5 detected ΔIC', f'{float(pw("半導體業", 5, .075)["mean_delta_vs_m0"]):+.3f}', '+0.014')
eq('H5 detected rate', f'{float(pw("半導體業", 5, .075)["reject_05_vs_m0"]):.0%}', '97%')
assert float(pw('金融保險業', 5, .075)['reject_05_vs_m0']) >= .97
eq('H20 ΔIC', f'{float(pw("半導體業", 20, .1)["mean_delta_vs_m0"]):+.3f}', '+0.017')
eq('H20 rate', f'{float(pw("半導體業", 20, .1)["reject_05_vs_m0"]):.0%}', '20%')
long_max = max(float(pw(i, h, .1)['mean_delta_vs_m0']) for i in industries for h in (30, 60))
assert long_max <= .009, long_max
checks.append({'label': 'H30/60 strongest injection mean ΔIC', 'value': f'≤ +0.009 (max {long_max:.4f})'})
for r in power:
    d = float(r['mean_delta_vs_m0']); h = r['horizon']
    if d >= eqj['delta_fl'][h]: assert float(r['excl_fl_vs_m0']) == 0, r
    if d >= .01: assert float(r['excl_001_vs_m0']) == 0, r
checks.append({'label': 'false exclusion of injected improvements', 'value': '0 in every cell reaching a bound'})
for v in ['4,800', '+0.014', '97%', '+0.017', '20%', '+0.009', '21 個種子', '0.25%', '0.17%', '0.05%', '0.5%']:
    assert v in eqtext, v

cp = js('docs/findings/25-cperm-summary.json')  # local-only in the research repo (LFS budget); thesis §9.5 carries the same numbers
mins = lambda null, k: min(cp['ranks'][b][null]['ridge'][k] for b in ['market', 'sector', 'allocation'])
eq('stratified ΔIC rank min', f'{mins("size_stratified", "min_p_rank_delta_ic"):.2f}', '0.09')
eq('unconditional ΔIC rank min', f'{mins("unconditional", "min_p_rank_delta_ic"):.2f}', '0.08')
eq('stratified R² rank min', f'{mins("size_stratified", "min_p_rank_r2"):.2f}', '0.19')
fp = cp['diagnostics']['market']['fixed_point_share_mean_max']
eq('self-mapping unconditional', f'{fp["unconditional"][0]:.1%}', '3.6%')
eq('self-mapping stratified', f'{fp["size_stratified"][0]:.1%}', '10.8%')
for v in ['0.09', '0.08', '0.19', '3.6%', '10.8%', '24 格']:
    assert v in text_of('#placebo > p'), v

dv = csvrows('docs/findings/25-design-variants-delta-ic.csv')
var = [r for r in dv if r['variant'] != 'base_ctrl']
eq('design variants', len({r['variant'] for r in var}), 11)
rm1 = [r for r in var if r['model'] == 'ridge' and r['candidate'] == 'M1']
eq('Ridge M1 variant cells negative', f'{sum(float(r["delta_ic"]) < 0 for r in rm1)}/{len(rm1)}', '88/88')
eq('all variant cells', len(var), 528)
eq('variant cells p<.05', sum(float(r['p_one_sided_gt0']) < .05 for r in var), 0)
dvs = js('docs/findings/25-design-variants-summary.json')
eq('main Ridge tunings at bound', dvs['base_ctrl']['ridge_alpha_counts'], {'1000': 2304})
rw = dvs['label_rank_wide']['ridge_alpha_counts']
eq('rank-label wide at 1e7', f'{rw["1e+07"]}/{sum(rw.values()):,}', '5/2,304')
eq('rank-label wide Ridge positive', dvs['label_rank_wide']['ridge/M1']['positive'], 0)
eq('U19 revert XGBoost positive', f'{dvs["u19_old"]["xgboost/M1"]["positive"]}/8', '6/8')
eq('XGBoost M0 IC current', f'{dvs["base_ctrl"]["xgboost/M1"]["mean_ic_m0"]:.3f}', '0.074')
eq('XGBoost M0 IC U19 reverted', f'{dvs["u19_old"]["xgboost/M1"]["mean_ic_m0"]:.3f}', '0.041')
fc = csvrows('docs/findings/25-fixed-composition-delta-ic.csv')
eq('fixed composition negative', f'{sum(float(r["delta_ic"]) < 0 for r in fc)}/{len(fc)}', '24/24')
eq('fixed semis stocks', sorted({r['n_stocks_fixed'] for r in fc if r['industry'] == '半導體業'}), ['18'])
eq('fixed financial stocks', sorted({r['n_stocks_fixed'] for r in fc if r['industry'] == '金融保險業'}), ['17'])
rob = text_of('#robustness > p')
for v in ['88', '528', '0.05', '18 檔', '17 檔', '24 格', '2,304', '5/2,304']:
    assert v in rob, v
mtext = text_of('#method > p')
for v in ['6/8', '0.074', '0.041', '0.020–0.071', '0.253–0.990']:
    assert v in mtext, v

p4 = js('docs/findings/25-p4-overlap-placebo.json')
p4r = csvrows('docs/findings/25-p4-overlap-placebo.csv')
eq('P4 over-rejection', f'{p4["reject_rate_range"][0]:.1%}–{p4["reject_rate_range"][1]:.1%}', '22.4%–60.4%')
eq('P4 rank p', f'{min(float(r["p_rank"]) for r in p4r):.2f}–{max(float(r["p_rank"]) for r in p4r):.2f}', '0.45–0.85')
p4row = [c for c in cells('verdict-table') if c[0] == 'P4'][0][3]
for v in ['22.4%–60.4%', '0.45–0.85', 'H=60']:
    assert v in p4row, v

perm_medians = [r['perm_median'] for r in inference['perm_rank']]
assert max(perm_medians) < 0
checks.append({'label': 'permutation ΔIC medians all negative', 'value': f'{min(perm_medians):.4f} to {max(perm_medians):.4f}'})
assert '22–36' in paper and '22–36' in text_of('#interpretation')
checks.append({'label': 'cross-section size', 'value': '22–36 (paper ch. 12–13)'})
assert '24 種組合' in text_of('#results .plain') and len(inference['perm_rank']) == 24
checks.append({'label': 'plain-language 24 combinations', 'value': '3 baselines × 2 industries × 4 horizons'})
assert '99 次' in text_of('#placebo .plain') and all(r['n_draws'] == 99 for r in inference['perm_rank'])
# The active-ETF qualifier in the interpretation block follows the data chapter.
assert '上市主動式（2025Q2 起）' in paper and '2025Q2 起另含少數主動式 ETF' in text_of('#interpretation')
# Paper wording for the precedent and its narrower novelty claim.
assert '未見以基準相對權重建構共同持股網路並在保留期檢驗其增量的先例' in paper
for name in ['Greenwood', 'Pavlova', 'Cremers', 'Jiang', 'Anton', 'Frazzini']:
    assert name in text_of('#precedent-table')
# Precedent block: every figure is quoted from the paper's §2.1 paragraph, not from memory.
for lit in ['1980–2003', '1.51%–2.40%', '1.13%–1.15%', '1984–2008', '7% 以上', '1983–2018', '2.8%–4.5%']:
    assert lit in paper and lit in text_of('#precedent'), lit
    checks.append({'label': f'precedent figure (paper §2.1)', 'value': lit})
assert '未見以基準相對權重建構共同持股網路並在保留期檢驗其增量的先例' in text_of('#precedent')

# ---- #30 review corrections, ahead of the thesis text: every number from findings/28 ----
rc = js('docs/findings/28-review-recompute.json')
nw, hh = rc['equivalence']['NW'], rc['equivalence']['HH']
eq('NW Bonferroni FL (recomputed)', f"{nw['excl_fl_bonf8']}/24", f"{excl('excl_fl_bonf8')}/24")
eq('HH Bonferroni FL', f"{hh['excl_fl_bonf8']}/24", '18/24')
eq('HH Bonferroni fixed', f"{hh['excl_fixed_bonf8']}/24", '16/24')
eq('NW Bonferroni fixed', f"{nw['excl_fixed_bonf8']}/24", '18/24')
eq('HH/NW SE ratio', '–'.join(f'{x:.2f}' for x in hh['se_ratio_hh_over_nw']), '1.14–1.26')
eq('HH market financials long UB', '–'.join(f'{x:.3f}' for x in hh['market_fin_long_ub_bonf8']), '0.017–0.028')
eq('HH not excluded', len(hh['not_excluded_fl_bonf8']), 6)
assert set(hh['not_excluded_fl_bonf8']) == {'market/半導體業/H60', 'market/金融保險業/H20', 'market/金融保險業/H30',
                                            'market/金融保險業/H60', 'sector/半導體業/H60', 'allocation/半導體業/H60'}
for v in ['23/24', '18/24', '16/24', '1.14–1.26', '六格']:
    assert v in text_of('#equivalence > .result-columns p'), v
assert '0.017–0.028' in text_of('#equivalence .note')
cwf = rc['cw_permutation_family']
eq('Ridge family-wise rejections (market)', cwf['market/ridge']['draws_with_any_bh_reject'], 16)
eq('XGBoost family-wise rejections (market)', cwf['market/xgboost']['draws_with_any_bh_reject'], 99)
eq('Ridge per-cell BH (market)', '–'.join(f'{x:.3f}' for x in cwf['market/ridge']['per_cell_bh']), '0.020–0.071')
for sel in ['#method > p', '#method details .note']:
    assert '16 次' in text_of(sel) and '未見明顯拒絕膨脹' not in text_of(sel), sel
p3m = rc['p3_matched']
eq('P3 matched valid-merge BH H=60', f"{p3m['bh_of_2x_mean_p'][3]:.3f}", '0.083')
assert not p3m['any_reject_valid'] and '0.083' in text_of('#placebo details') and '500' in text_of('#placebo details')

# ---- Features block: counts quoted from paper ch. 6, ablation and permutation figures from findings/18 ----
ftext = text_of('#features')
for lit in ['45 個', '實際 44 個', '0.90', 'VIF', '41.91–42.00', '36–43']:
    assert lit in ftext, lit
for lit in ['41.91–42.00', '36–43 個區間', 'M0 恆為 20.00 個', '絕對相關高於 0.90', 'VIF $\\geq 10$']:
    assert lit in paper, lit
    checks.append({'label': 'feature count/screen (paper ch. 6)', 'value': lit})
assert 'M0 固定 20 個' in ftext
# 'main' repeats the market baseline; count the three baselines once each.
three = [r for r in inference['delta_ic'] if r['source'] in ('market', 'sector', 'allocation')]
only_etf = [r for r in three if r['candidate'] == 'M1_noPrice']
only_price = [r for r in three if r['candidate'] == 'M1_noETF']
assert len(only_etf) == len(only_price) == 24
eq('ETF-track-only positive cells', f"{sum(r['delta_ic'] > 0 for r in only_etf)}/{len(only_etf)}", '4/24')
assert all(r['source'] == 'market' and r['industry'] == '金融保險業' for r in only_etf if r['delta_ic'] > 0)
eq('price-track-only positive cells', sum(r['delta_ic'] > 0 for r in only_price), 0)
pm = [r['perm_median'] for r in inference['perm_rank']]
eq('permutation ΔIC median range', f'{min(pm):.3f} 至 {max(pm):.3f}'.replace('-', '−'), '−0.034 至 −0.008')
for lit in ['+0.0016 至 +0.0039', '−0.034 至 −0.008', '2,304', '24 格中只有市場基準的金融四格']:
    assert lit in ftext, lit
for name, year in [('Amihud', 2002), ('Cohen', 2008), ('Ying', 2024)]:
    assert f'{name}' in ftext and str(year) in ftext

# Audit: every number in a block written or rewritten for #28 is sourced above or structural.
rendered28 = {t.replace(',', '') for t in re.findall(r'\d+(?:[.,]\d+)*', ' '.join(str(c['value']) + ' ' + c['label'] for c in checks))}
structural28 = {'2025', '5', '20', '30', '60', '8', '95', '0.25', '252', '0.01', '1989', '2008', '2009', '2014', '2021', '2023', '2026', '2020', '2007',
                '0050', '0053', '0056', '19', '1', '2', '8.6', '9.5', '10.11', '2.1', '5.1', '11.1', '11.5', '12', 'D.13', '13', '15', '16', '99', '22', '36'}
audit = ['.hero .gist', '.plain', '#equivalence p', '#equivalence .note', '#robustness > p', '#interpretation', '#method > p', '#cperm']
for sel in audit:
    for token in re.findall(r'\d+(?:[.,]\d+)*', text_of(sel)):
        assert token.replace(',', '') in rendered28 or token in structural28, f'untraced number in {sel}: {token}'
for token in re.findall(r'\d+(?:[.,]\d+)*', p4row + ' ' + text_of('#gat-text')):
    assert token.replace(',', '') in rendered28 or token in structural28, token
checks.append({'label': 'issue-28 number audit', 'value': 'every numeric token in new or rewritten blocks sourced or structural'})

visible = soup.get_text(' ', strip=True)
# Literature entries must exactly match the paper; no unused list entries.
for li in soup.select('.references li'):
    text = li.get_text(' ', strip=True)
    expected = next(line for line in paper.splitlines() if line.startswith(text.split(' (')[0]))
    eq('APA reference', text, expected.replace('*', ''))
for name, year in [('Amihud',2002),('Cohen',2008),('Ying',2024),('Frazzini',2016),('Anton',2021),('Cremers',2009),('Greenwood',2008),('Grinold',1989),('Jiang',2014),('Pavlova',2023),('Anton',2014),('Benjamini',1995),('Brown',2021),('Campbell',2008),('Clark',2007),('Escanciano',2026),('Fama',1973),('Gu',2020)]:
    assert re.search(rf'{name}.{{0,45}}{year}', visible)
    assert any(name in li.get_text() for li in soup.select('.references li'))
assert '© 2026 黃晨瑋、鐘珮甄、黃苡晴' in visible
assert 'alloc_debiasing/' not in str(soup)
for leak in ['/Users/', 'file://', 'data/tej', 'data/pipeline', '.venv', 'scratchpad']:
    assert leak not in str(soup), leak  # no local paths or licensed raw-data locations on the public page
assert '登錄' not in visible and '註冊' not in visible
assert 'ETF 重疊梯度' in soup.select_one('#verdict-table').get_text()
for literal in ['74.7%', '15.2%']:
    assert literal in paper and literal in visible
checks.append({'label':'gate pass rates', 'value':'paper §9.9: 74/99, 15/99; units are draws, not cells'})

research_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
manifest = {'research_commit': research_sha, 'scope':'Aggregate export consistency; no model rerun, raw TEJ access, or provenance certification',
            'source_sha256_lf':sources, 'check_count':len(checks), 'checks':checks}
(poster/args.out/'evidence.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'{len(checks)} evidence checks passed across {len(sources)} aggregate/paper sources.')
