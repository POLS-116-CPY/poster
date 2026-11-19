// npm install --no-save playwright; node scripts/export_print.cjs [out-dir]
// Renders print/zenkai.html to poster.pdf (全開 787 × 1092 mm, portrait) and fails on any layout or PDF defect.
// PW_CHROMIUM may point at an already-installed Chromium when the bundled build is absent.
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');

const MM = 96 / 25.4;
const PAGE = { w: 787, h: 1092 };
// Floors for reading at arm's length; figure labels are scaled SVG text, so they are measured as rendered.
const FLOOR_PT = { body: 18, table: 14, note: 12, figure: 11 };

(async () => {
  const root = path.resolve(__dirname, '..');
  const out = path.resolve(process.argv[2] || path.join(root, 'review', 'issue-29'));
  fs.mkdirSync(out, { recursive: true });
  const pdfPath = path.join(root, 'poster.pdf');
  const browser = await chromium.launch({ headless: true, executablePath: process.env.PW_CHROMIUM || undefined });
  const report = {};
  try {
    const page = await browser.newPage({ viewport: { width: Math.round(PAGE.w * MM), height: Math.round(PAGE.h * MM) } });
    await page.emulateMedia({ media: 'print' });
    await page.goto('file://' + path.join(root, 'print', 'zenkai.html'), { waitUntil: 'networkidle' });
    await page.evaluate(() => document.fonts.ready);
    const m = await page.evaluate(({ MM }) => {
      const pt = px => +(px * 72 / 96).toFixed(2);
      // Serif is only used at weight 900 (title, headings); Google serves unicode-range subsets, so test real glyphs.
      const fonts = [['Noto Sans TC', '400'], ['Noto Sans TC', '700'], ['Noto Serif TC', '900']].map(([f, w]) => [`${f} ${w}`, document.fonts.check(`${w} 16px "${f}"`, '去偏網路')]);
      // Measure natural height: with space-between the last section always touches the bottom.
      document.querySelectorAll('.col').forEach(c => { c.style.justifyContent = 'flex-start'; });
      const cols = [...document.querySelectorAll('.col')].map((c, i) => {
        const box = c.getBoundingClientRect();
        const bottom = Math.max(...[...c.children].map(k => k.getBoundingClientRect().bottom));
        // Negative slack means the column runs past its grid cell, which print would clip.
        return { col: i + 1, height_mm: +(box.height / MM).toFixed(1), slack_mm: +((box.bottom - bottom) / MM).toFixed(1) };
      });
      document.querySelectorAll('.col').forEach(c => { c.style.justifyContent = ''; });
      const pageEl = document.querySelector('.page');
      const pageOverflow = pageEl.scrollHeight - pageEl.clientHeight;
      const minOf = sel => Math.min(...[...document.querySelectorAll(sel)].filter(e => e.textContent.trim()).map(e => parseFloat(getComputedStyle(e).fontSize)));
      const svgs = [...document.querySelectorAll('main svg')];
      const figureMin = Math.min(...svgs.flatMap(svg => [...svg.querySelectorAll('text')].map(t => {
        const scale = svg.getBoundingClientRect().width / svg.viewBox.baseVal.width;
        return parseFloat(getComputedStyle(t).fontSize) * scale;
      })));
      const clipped = svgs.flatMap((svg, i) => [...svg.querySelectorAll('text')].flatMap(t => {
        const b = t.getBBox(), v = svg.viewBox.baseVal;
        const mtx = svg.getScreenCTM().inverse().multiply(t.getScreenCTM());
        const pts = [[b.x, b.y], [b.x + b.width, b.y + b.height]].map(([x, y]) => new DOMPoint(x, y).matrixTransform(mtx));
        return pts.some(p => p.x < v.x - 1 || p.y < v.y - 1 || p.x > v.x + v.width + 1 || p.y > v.y + v.height + 1) ? [{ svg: i, text: t.textContent }] : [];
      }));
      // Text boxes that leave their column would be cut off or collide with the neighbour.
      const escapes = [...document.querySelectorAll('.col *')].filter(e => {
        const c = e.closest('.col').getBoundingClientRect(), b = e.getBoundingClientRect();
        return b.width && (b.right > c.right + 1 || b.left < c.left - 1);
      }).map(e => e.tagName + ':' + e.textContent.trim().slice(0, 30));
      return {
        fonts, cols, pageOverflow, svgCount: svgs.length, clipped, escapes: escapes.slice(0, 10),
        // Header prose (verdict, gist, definitions, author and QR lines) is read at the same distance as the columns.
        min_pt: { body: pt(minOf('.c p, .hero p, .hero .meta > div')), table: pt(minOf('.c td, .c th')), note: pt(minOf('.note, caption')), figure: pt(figureMin) },
      };
    }, { MM });
    report.layout = m;
    if (process.env.MEASURE) { console.log(JSON.stringify(m, null, 2)); await page.screenshot({ path: path.join(out, 'measure.png') }); return; }
    for (const [f, ok] of m.fonts) assert(ok, `font not loaded: ${f}`);
    assert.equal(m.pageOverflow, 0, 'page content overflows');
    for (const c of m.cols) assert(c.slack_mm >= 0, `column ${c.col} overflows by ${-c.slack_mm} mm`);
    assert.equal(m.svgCount, 4, 'expected 4 figures in the poster body');
    assert.deepEqual(m.clipped, [], 'clipped SVG labels');
    assert.deepEqual(m.escapes, [], 'elements wider than their column');
    for (const k of Object.keys(FLOOR_PT)) assert(m.min_pt[k] >= FLOOR_PT[k], `${k} text ${m.min_pt[k]} pt < ${FLOOR_PT[k]} pt`);

    await page.pdf({ path: pdfPath, preferCSSPageSize: true, printBackground: true });
  } finally {
    await browser.close();
  }

  const info = execFileSync('pdfinfo', [pdfPath], { encoding: 'utf8' });
  const pages = +info.match(/Pages:\s+(\d+)/)[1];
  const [w, h] = info.match(/Page size:\s+([\d.]+) x ([\d.]+)/).slice(1).map(Number);
  assert.equal(pages, 1, `PDF has ${pages} pages`);
  assert(Math.abs(w / 72 * 25.4 - PAGE.w) < 1 && Math.abs(h / 72 * 25.4 - PAGE.h) < 1, `page is ${w} x ${h} pt`);
  const fonts = execFileSync('pdffonts', [pdfPath], { encoding: 'utf8' });
  for (const f of ['NotoSansTC', 'NotoSerifTC']) assert(fonts.includes(f), `${f} not in PDF`);
  // pdffonts column "emb": every font line must say yes, otherwise the printer substitutes glyphs.
  const notEmbedded = fonts.split('\n').slice(2).filter(l => l.trim() && !/\byes\s+(yes|no)\s+(yes|no)\s+\d+/.test(l));
  assert.deepEqual(notEmbedded, [], 'fonts not embedded');
  // A system-font fallback (PingFang, Helvetica) means a glyph is missing from the Noto stack.
  const foreign = [...new Set(fonts.split('\n').slice(2).map(l => l.split(/\s+/)[0].replace(/^[A-Z]+\+/, '')).filter(f => f && !f.startsWith('Noto')))];
  assert.deepEqual(foreign, [], 'non-Noto fonts in PDF');
  // Selectable text proves the page was not rasterised.
  const text = execFileSync('pdftotext', [pdfPath, '-'], { encoding: 'utf8' });
  assert(text.includes('未建立增量'), 'PDF text is not selectable');

  report.pdf = { pages, size_mm: [+(w / 72 * 25.4).toFixed(1), +(h / 72 * 25.4).toFixed(1)] };
  fs.writeFileSync(path.join(out, 'export.json'), JSON.stringify(report, null, 2) + '\n');
  console.log(JSON.stringify(report, null, 2));
})().catch(e => { console.error(e.message); process.exit(1); });
