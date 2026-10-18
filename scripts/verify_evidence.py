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
