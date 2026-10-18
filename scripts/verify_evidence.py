"""Verify the public poster against aggregate exports, never raw TEJ data.

python scripts/verify_evidence.py --research-root /path/to/POLS-116-CPY
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
for row, (industry, h) in enumerate((i, h) for i in industries for h in horizons):
    r = next(r for r in decomp if r['model'] == 'ridge' and r['industry'] == industry and int(r['horizon']) == h)
    for col, metric in enumerate(['r2_pooled', 'r2_cross_sectional', 'r2_cs_net_of_shrink'], 1):
        eq(f'Ridge/{industry}/{h}/{metric}', cells('ridge-table')[row][col], signed(float(r[metric])*100)+'%')

for bar in soup.select('#decomposition-chart rect[data-value]'):
    r = next(r for r in decomp if r['model'] == 'xgboost' and r['industry'] == bar['data-industry'] and r['horizon'] == bar['data-horizon'])
    v = float(r[bar['data-metric']])
    near('XGBoost bar data', bar['data-value'], v)
    near('XGBoost bar origin', bar['x'], 330 + min(0, v*100)*95)
    near('XGBoost bar width', bar['width'], abs(v)*100*95)
    eq('XGBoost visible label', bar.find_next_sibling('text').get_text(), signed(v*100)+'%')

mde = csvrows('results/15_power/mde.csv')
for row, h in enumerate(horizons):
    for col, industry in enumerate(industries, 1):
        r = next(r for r in mde if r['sample'] == industry and int(r['horizon']) == h and r['threshold'] == 'BH_most_conservative')
        eq(f'MDE/{industry}/{h}', cells('mde-table')[row][col], f'{float(r["mde_pct"]):.2f}%')

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

gat = csvrows('results/13_gat/summary.csv')
for row, industry in enumerate(industries):
    for col, h in enumerate(horizons, 1):
        r = next(r for r in gat if r['industry'] == industry and r['horizon'] == str(h) and r['baseline'] == 'M0')
        eq(f'GAT/{industry}/{h}', cells('gat-table')[row][col], f'{float(r["r2_oos"])*100:.1f}%')

fm = csvrows('results/18_fama_macbeth/fama_macbeth.csv')
for index, industry in enumerate(industries):
    for col, h in enumerate(horizons, 1):
        r = next(r for r in fm if r['industry'] == industry and r['horizon'] == str(h) and r['model'] == 'ridge' and r['variant'] == 'M1_incremental')
        eq(f'FM coefficient/{industry}/{h}', cells('fm-table')[index*2][col], f'{float(r["gamma_mean"]):.2f}')
        eq(f'FM p/{industry}/{h}', cells('fm-table')[index*2+1][col], f'{float(r["nw_p"]):.2f}')

for row, baseline in enumerate(['market', 'sector', 'allocation']):
    b = evidence['baselines'][baseline]
    groups = [b['total']] + [b['groups'][g] for g in ['substantive_including_base', 'permute', 'noise']]
    for col, group in enumerate(groups, 1):
        eq(f'robustness/{baseline}/{col}', cells('robustness-table')[row][col], f'{group["positive_cells"]}/{group["cells"]}')
    for col, model in enumerate(['ridge', 'xgboost'], 5):
        rates = [r['reject_rate'] for r in b['placebo'] if r['model'] == model]
        eq(f'CW range/{baseline}/{model}', cells('robustness-table')[row][col], f'{min(rates):.3f}–{max(rates):.3f}')

variants = csvrows('results/sweep/variant_results_full.csv')
for row, (industry, variant) in enumerate((i, v) for i in industries for v in ['base', 'etf_cap_broad']):
    for col, h in enumerate(horizons, 1):
        r = next(r for r in variants if r['variant'] == variant and r['industry'] == industry and r['horizon'] == str(h) and r['model'] == 'xgboost' and r['spec'] == 'M1' and r['baseline'] == 'M0')
        eq(f'ETF/{industry}/{variant}/{h}', cells('etf-table')[row][col], signed(float(r['r2_oos'])*100)+'%')

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

alpha = js('docs/findings/19-ridge-wide-alpha-counts.json')
eq('main logged upper-bound choices', alpha['main_spec']['count_at_max_alpha'], 216)
eq('actual main tuning count', alpha['main_spec']['total_tuned_retrains_n_tuned'], 2304)
eq('wide upper-bound choices', alpha['ridge_wide']['count_at_max_alpha'], 180)
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
residuals = {tr['data-code']: [td.get_text(strip=True) for td in tr.find_all('td')] for tr in soup.select('#debias-residuals tr[data-code]')}
assert list(residuals) == list(ex_stocks)
for code, s in ex_stocks.items():
    for k, b in enumerate(['market', 'sector', 'allocation'], 1):
        eq(f'example residual/{b}/{code}', residuals[code][k], ex_vec(s[f'residual_{b}']))
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
cos_rows = {tr['data-pair']: tr for tr in soup.select('#cosine-table tr[data-pair]')}
assert list(cos_rows) == [f'{i}-{j}' for i, j in ex_cos]
for (i, j), r in ex_cos.items():
    tds = cos_rows[f'{i}-{j}'].find_all('td')
    eq(f'cosine pair/{i}-{j}', tds[0].get_text(strip=True), f'{ex_names[i]}–{ex_names[j]}')
    for k, b in enumerate(baselines3, 1):
        eq(f'cosine/{b}/{i}-{j}', tds[k].get_text(strip=True), ex_signed(r[b], 3))
        assert ('na' in (tds[k].get('class') or [])) == (abs(r[b]) < .5), (i, j, b)
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
for phrase in ['教學示例，非估計結果', '持股取 MOPS 2023Q4', '亦取 2023Q4', '兩季前公告的持股', '只以圖中三檔 ETF 為維度']:
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

paper = read('docs/paper/paper.md')
visible = soup.get_text(' ', strip=True)
# Literature entries must exactly match the paper; no unused list entries.
for li in soup.select('.references li'):
    text = li.get_text(' ', strip=True)
    expected = next(line for line in paper.splitlines() if line.startswith(text.split(' (')[0]))
    eq('APA reference', text, expected.replace('*', ''))
for name, year in [('Anton',2014),('Benjamini',1995),('Brown',2021),('Campbell',2008),('Clark',2007),('Escanciano',2026),('Fama',1973),('Gu',2020)]:
    assert re.search(rf'{name}.{{0,45}}{year}', visible)
    assert any(name in li.get_text() for li in soup.select('.references li'))
assert '© 2026 黃晨瑋、鐘珮甄、黃苡晴' in visible
assert 'alloc_debiasing/' not in str(soup)
assert '登錄' not in visible and '註冊' not in visible
assert 'ETF 重疊梯度' in soup.select_one('#verdict-table').get_text()
for literal in ['74.7%', '15.2%']:
    assert literal in paper and literal in visible
checks.append({'label':'gate pass rates', 'value':'paper §9.9: 74/99, 15/99; units are draws, not cells'})

research_sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
manifest = {'research_commit': research_sha, 'scope':'Aggregate export consistency; no model rerun, raw TEJ access, or provenance certification',
            'source_sha256_lf':sources, 'check_count':len(checks), 'checks':checks}
(poster/'review/issue-22/evidence.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(f'{len(checks)} evidence checks passed across {len(sources)} aggregate/paper sources.')
