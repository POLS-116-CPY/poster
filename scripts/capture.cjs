// npm install --no-save playwright; node scripts/capture.cjs before|after
const { chromium } = require('playwright');
const fs = require('node:fs');
const path = require('node:path');
const http = require('node:http');
const assert = require('node:assert/strict');
const { execFileSync } = require('node:child_process');

(async () => {
  const phase = process.argv[2] || 'after';
  assert(['before', 'after'].includes(phase));
  const root = path.resolve(__dirname, '..');
  const out = path.join(root, 'review', 'issue-22');
  fs.mkdirSync(out, { recursive: true });
  // Keep the comparison baseline fixed after the revision itself is committed.
  const baseline = 'a84930137d9a90b5d51f7912750c41777d21f0f7';
  const html = phase === 'before' ? execFileSync('git', ['show', `${baseline}:index.html`], {cwd:root}) : fs.readFileSync(path.join(root,'index.html'));
  const server = http.createServer((req, res) => {
    res.setHeader('Content-Type', 'text/html; charset=utf-8');
    res.end(html);
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  let browser;
  const checks = [];
  try {
    browser = await chromium.launch({ headless: true });
    for (const colorScheme of ['light', 'dark']) {
      for (const width of [1440, 400, 700, 900]) {
        const page = await browser.newPage({ viewport: { width, height: 1000 }, colorScheme, deviceScaleFactor: 1 });
        await page.goto(`http://127.0.0.1:${server.address().port}`, { waitUntil: 'networkidle' });
        await page.evaluate(() => document.fonts.ready);
        const collapsedHeight = await page.evaluate(() => document.documentElement.scrollHeight);
        if ([1440,400].includes(width)) await page.screenshot({ path:path.join(out,`${phase}-${width}-${colorScheme}.png`), fullPage:true });
        // Test every diagram and table, including content disclosed on demand.
        if (phase === 'after') await page.locator('details').evaluateAll(ds => ds.forEach(d => d.open = true));
        const check = await page.evaluate(() => {
          const svgs = [...document.querySelectorAll('svg')];
          const clipped = svgs.flatMap((svg, i) => [...svg.querySelectorAll('text')].flatMap(t => {
            const b = t.getBBox(), v = svg.viewBox.baseVal;
            // getBBox is in the text's local coordinates; transform to root SVG space.
            const m = svg.getScreenCTM().inverse().multiply(t.getScreenCTM());
            const pts = [[b.x,b.y],[b.x+b.width,b.y],[b.x,b.y+b.height],[b.x+b.width,b.y+b.height]].map(([x,y]) => new DOMPoint(x,y).matrixTransform(m));
            return pts.some(p => p.x < v.x - 1 || p.y < v.y - 1 || p.x > v.x+v.width+1 || p.y > v.y+v.height+1) ? [{svg:i, text:t.textContent}] : [];
          }));
          const textOverlaps = svgs.flatMap((svg, i) => {
            const texts = [...svg.querySelectorAll('text')].map(t => ({t, b:t.getBoundingClientRect()}));
            const overlaps = [];
            for (let a=0;a<texts.length;a++) for(let b=a+1;b<texts.length;b++) {
              const A=texts[a],B=texts[b];
              const x=Math.min(A.b.right,B.b.right)-Math.max(A.b.left,B.b.left);
              const y=Math.min(A.b.bottom,B.b.bottom)-Math.max(A.b.top,B.b.top);
              if(x>1 && y>1) overlaps.push({svg:i,a:A.t.textContent,b:B.t.textContent});
            }
            return overlaps;
          });
          const rgba = s => { const a=s.match(/[\d.]+/g)?.map(Number); return a ? [a[0],a[1],a[2],a[3] ?? 1] : [0,0,0,0]; };
          const blend = (f,b) => f.slice(0,3).map((v,i)=>v*f[3]+b[i]*(1-f[3]));
          const luminance = c => c.map(v=>v/255).map(v=>v<=.04045?v/12.92:((v+.055)/1.055)**2.4).reduce((s,v,i)=>s+v*[.2126,.7152,.0722][i],0);
          const contrastFailures = [], ratios = [];
          const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
          let node;
          while ((node=walker.nextNode())) {
            const e=node.parentElement;
            if(!node.textContent.trim() || ['SCRIPT','STYLE','TITLE','DESC'].includes(e.tagName.toUpperCase())) continue;
            const range=document.createRange(); range.selectNodeContents(node);
            const rect=range.getBoundingClientRect();
            if(!rect.width || !rect.height) continue;
            const style=getComputedStyle(e), svg=e.closest('svg');
            const chain=[]; for(let p=e;p;p=p.parentElement) chain.unshift(p);
            let bg=[255,255,255]; for(const p of chain) bg=blend(rgba(getComputedStyle(p).backgroundColor),bg);
            if(svg) {
              const center=new DOMPoint(rect.x+rect.width/2,rect.y+rect.height/2).matrixTransform(svg.getScreenCTM().inverse());
              for(const r of svg.querySelectorAll('rect')) {
                if(!(r.compareDocumentPosition(e)&Node.DOCUMENT_POSITION_FOLLOWING)) continue;
                const b=r.getBBox();
                if(center.x>=b.x && center.x<=b.x+b.width && center.y>=b.y && center.y<=b.y+b.height) {
                  const rs=getComputedStyle(r), fill=rgba(rs.fill); fill[3]*=Number(rs.fillOpacity)*Number(rs.opacity); bg=blend(fill,bg);
                }
              }
            }
            const fg=rgba(svg?style.fill:style.color); fg[3]*=Number(style.opacity);
            const l1=luminance(blend(fg,bg)),l2=luminance(bg),ratio=(Math.max(l1,l2)+.05)/(Math.min(l1,l2)+.05);
            const fontSize=parseFloat(style.fontSize),bold=Number(style.fontWeight)>=700;
            const required=fontSize>=24 || (bold && fontSize>=18.667)?3:4.5;
            ratios.push(ratio);
            if(ratio<required-.005) contrastFailures.push({text:node.textContent.trim().slice(0,90),ratio:Number(ratio.toFixed(3)),required});
          }
          return { pageWidth: document.documentElement.clientWidth, scrollWidth: document.documentElement.scrollWidth,
            height: document.documentElement.scrollHeight, svgCount:svgs.length, tableCount:document.querySelectorAll('table').length,
            clipped, textOverlaps, minTextContrast:Number(Math.min(...ratios).toFixed(3)),contrastFailures,font:document.fonts.check('15px "Noto Sans TC"'),
            sections:[...document.querySelectorAll('.c')].map(c=>({title:c.querySelector('h2')?.textContent.trim(),top:Math.round(c.getBoundingClientRect().top),height:Math.round(c.getBoundingClientRect().height)})) };
        });
        checks.push({width,colorScheme,collapsedHeight,...check});
        if (phase === 'after' && [1440,400].includes(width)) await page.screenshot({ path:path.join(out,`${phase}-expanded-${width}-${colorScheme}.png`), fullPage:true });
        await page.close();
      }
    }
    fs.writeFileSync(path.join(out, `${phase}-layout.json`), JSON.stringify(checks,null,2)+'\n');
    console.log(JSON.stringify(checks.map(({sections,contrastFailures,...c})=>({...c,contrastFailures:contrastFailures.slice(0,8)})),null,2));
    if(phase==='after') for(const c of checks) {
      assert(c.scrollWidth <= c.width, `Page overflow: ${c.width} ${c.colorScheme}`);
      assert.equal(c.clipped.length,0,`Clipped SVG: ${c.width} ${c.colorScheme}`);
      assert.equal(c.textOverlaps.length,0,`Overlapping SVG labels: ${c.width} ${c.colorScheme}`);
      assert.equal(c.contrastFailures.length,0,`WCAG AA text contrast: ${c.width} ${c.colorScheme}`);
    }
  } finally { if(browser) await browser.close(); server.close(); }
})().catch(e=>{console.error(e);process.exitCode=1;});
