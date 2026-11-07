// Usage: node scripts/export-workflow-svg.mjs <diagram.html> <out-dark.svg> <out-light.svg> [--loop <node-id>,...]
// Opens the Archify HTML in headless Chrome and writes a standalone SVG per theme,
// with computed styles inlined so it renders as a plain <img> (e.g. on GitHub).
// --loop draws a repeat arrow under each listed node, which Archify cannot route.
import { spawn } from 'node:child_process';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const args = process.argv.slice(2);
const loopAt = args.indexOf('--loop');
const loops = loopAt < 0 ? [] : args.splice(loopAt, 2)[1].split(',');
const [input, outDark, outLight] = args;
const chromePath = process.env.CHROME || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const profile = fs.mkdtempSync(path.join(os.tmpdir(), 'svg-export-'));
const chrome = spawn(chromePath, [
  '--headless=new', '--disable-gpu', '--no-first-run', '--remote-debugging-port=0',
  `--user-data-dir=${profile}`, '--window-size=2048,1400', 'about:blank',
], { stdio: 'ignore' });

const sleep = ms => new Promise(r => setTimeout(r, ms));
let port;
for (let i = 0; i < 100 && !port; i++) {
  await sleep(100);
  try { port = fs.readFileSync(path.join(profile, 'DevToolsActivePort'), 'utf8').split('\n')[0]; } catch {}
}
const targets = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
const ws = new WebSocket(targets.find(t => t.type === 'page').webSocketDebuggerUrl);
await new Promise(r => ws.addEventListener('open', r, { once: true }));
let seq = 0;
const pending = new Map();
ws.addEventListener('message', e => {
  const msg = JSON.parse(e.data);
  if (msg.id && pending.has(msg.id)) { pending.get(msg.id)(msg); pending.delete(msg.id); }
});
const send = (method, params = {}) => new Promise(r => { const id = ++seq; pending.set(id, r); ws.send(JSON.stringify({ id, method, params })); });
const evaluate = async expression => {
  const res = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
  if (res.result.exceptionDetails) throw new Error(JSON.stringify(res.result.exceptionDetails));
  return res.result.result.value;
};

await send('Page.enable');
await send('Page.navigate', { url: pathToFileURL(path.resolve(input)).href });
await sleep(2500);

const exportTheme = theme => evaluate(`(async () => {
  const root = document.documentElement;
  root.dataset.theme = ${JSON.stringify(theme)};
  root.dataset.motion = 'still';
  const container = document.querySelector('.diagram-container');
  if (container) container.dataset.detailLevel = 'full';
  await new Promise(r => setTimeout(r, 400));
  const svg = document.querySelector('.diagram-container > svg') || document.querySelector('svg');
  const props = ['fill','fill-opacity','stroke','stroke-width','stroke-opacity','stroke-dasharray',
    'stroke-linecap','stroke-linejoin','opacity','font-family','font-size','font-weight',
    'letter-spacing','text-anchor','dominant-baseline','paint-order','visibility'];
  const clone = svg.cloneNode(true);
  const src = [svg, ...svg.querySelectorAll('*')];
  const dst = [clone, ...clone.querySelectorAll('*')];
  const drop = [];
  src.forEach((el, i) => {
    const cs = getComputedStyle(el);
    const out = dst[i];
    if (cs.display === 'none' || cs.visibility === 'hidden' || ['script','foreignObject','animate','animateMotion','set'].includes(el.tagName)) { drop.push(out); return; }
    out.setAttribute('style', props.map(p => p + ':' + cs.getPropertyValue(p)).join(';'));
    out.removeAttribute('class');
    for (const a of [...out.attributes]) if (a.name.startsWith('data-') || a.name.startsWith('aria-') || a.name === 'tabindex' || a.name === 'role') out.removeAttribute(a.name);
  });
  const ns = 'http://www.w3.org/2000/svg';
  for (const id of ${JSON.stringify(loops)}) {
    const node = svg.querySelector('[data-node-id="' + id + '"]');
    if (!node) throw new Error('No node ' + id);
    const box = [...node.querySelectorAll('rect')].filter(r => getComputedStyle(r).stroke !== 'none').map(r => [r, r.getBBox()])
      .sort((a, b) => b[1].width * b[1].height - a[1].width * a[1].height)[0];
    const tag = [...node.querySelectorAll('text')].find(t => t.textContent === node.dataset.nodeTag);
    const color = tag ? getComputedStyle(tag).fill : getComputedStyle(box[0]).stroke;
    const cx = box[1].x + box[1].width / 2, bottom = box[1].y + box[1].height;
    const arc = document.createElementNS(ns, 'path');
    arc.setAttribute('d', 'M ' + (cx + 6) + ' ' + (bottom + 1) + ' A 9 9 0 1 1 ' + (cx - 6) + ' ' + (bottom + 6));
    arc.setAttribute('style', 'fill:none;stroke:' + color + ';stroke-width:1.8px;stroke-linecap:round');
    const head = document.createElementNS(ns, 'path');
    head.setAttribute('d', 'M ' + (cx - 6) + ' ' + (bottom + 0.5) + ' l -4 6.2 h 8 z');
    head.setAttribute('style', 'fill:' + color + ';stroke:none');
    const out = dst[src.indexOf(node)];
    out.append(arc, head);
  }
  drop.forEach(el => el.remove());
  clone.querySelectorAll('style').forEach(el => el.remove());
  clone.setAttribute('xmlns', 'http://www.w3.org/2000/svg');
  clone.removeAttribute('width'); clone.removeAttribute('height');
  const [, , w, h] = clone.getAttribute('viewBox').split(/[ ,]+/).map(Number);
  clone.setAttribute('width', w); clone.setAttribute('height', h);
  const bg = getComputedStyle(document.body).backgroundColor;
  const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
  rect.setAttribute('width', '100%'); rect.setAttribute('height', '100%'); rect.setAttribute('fill', bg);
  clone.insertBefore(rect, clone.firstChild);
  return new XMLSerializer().serializeToString(clone);
})()`);

fs.writeFileSync(outDark, await exportTheme('dark') + '\n');
fs.writeFileSync(outLight, await exportTheme('light') + '\n');
ws.close();
chrome.kill();
