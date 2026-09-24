// Headless check that the brain still produces the three behaviors.
// Usage: node data/behavior_test.cjs [path/to/index.html]
// Needs Playwright (npm i -D playwright) and a Chromium it can launch.
const { chromium } = require('playwright');
const { resolve } = require('node:path');

(async () => {

const file = resolve(process.argv[2] || resolve(__dirname, '../index.html'));
const browser = await chromium.launch();
const page = await browser.newPage();
await page.goto('file://' + file);
await page.waitForTimeout(300);

const overrides = JSON.parse(process.env.SYN_OVERRIDES || '{}'); // e.g. {"antL>dnL": 4}
const bias = JSON.parse(process.env.BIAS_OVERRIDES || '{}'); // e.g. {"mb": -1}
const res = await page.evaluate(([overrides, bias, SCALE]) => {
  for (const s of SYN) { const k = s[0] + '>' + s[1]; if (k in overrides) s[2] = overrides[k]; }
  Object.assign(BIAS, bias);
  const scale = JSON.parse(SCALE); for (const s of SYN) if (s[3] in scale) s[2] *= scale[s[3]];
  window.requestAnimationFrame = () => 0; // stop the real-time loop; we step by hand
  window.send = () => {};
  const dt = 1 / 60;
  let escapes = 0;
  const origEscape = escape;
  // count escapes without touching the page's own function binding
  function step() {
    updateThreat(dt);
    const bo = brainStep(dt);
    S.escapeCd = Math.max(0, S.escapeCd - dt);
    if (S.reflex && bo.gf > 0.6 && S.escapeCd === 0) { escapes++; origEscape(); }
    computeCmd(bo); physics(dt);
  }
  function reset(x, y, h) {
    Object.assign(S.fly, { x, y, h, vx: 0, vy: 0, alt: 30, spd: 0 });
    Object.assign(S, { armed: true, flying: true, landing: false, auto: true, odor: null, light: null, threat: null, bat: 100, escapeCd: 0 });
    for (const k in A) A[k] = 0;
    escapes = 0;
  }
  const dist = p => Math.hypot(S.fly.x - p.x, S.fly.y - p.y);
  const out = {};
  // 1) odor and 2) light: fly starts at the center facing up; target placed in 6 directions
  const targets = [{ x: 100, y: 20 }, { x: 20, y: 20 }, { x: 100, y: 55 }, { x: 20, y: 55 }, { x: 60, y: 62 }, { x: 110, y: 35 }];
  for (const kind of ['odor', 'light']) {
    const r = [];
    for (let trial = 0; trial < 3; trial++) for (const t of targets) {
      reset(60, 35, -Math.PI / 2);
      S[kind] = { ...t };
      const d0 = dist(t); let best = d0, reachedAt = null;
      for (let i = 0; i < 60 * 25; i++) {
        step(); const d = dist(t); best = Math.min(best, d);
        if (d < 8 && reachedAt === null) reachedAt = i * dt;
      }
      r.push({ d0: +d0.toFixed(0), best: +best.toFixed(1), final: +dist(t).toFixed(1), reachedAt, escapes });
    }
    out[kind] = { reached: r.filter(x => x.reachedAt !== null).length + '/' + r.length,
      meanTime: +(r.filter(x => x.reachedAt !== null).reduce((a, x) => a + x.reachedAt, 0) / Math.max(1, r.filter(x => x.reachedAt !== null).length)).toFixed(1),
      meanFinal: +(r.reduce((a, x) => a + x.final, 0) / r.length).toFixed(1),
      falseEscapes: r.reduce((a, x) => a + x.escapes, 0) };
  }
  // 3) threat: looming from random directions
  let hit = 0, lat = [];
  for (let trial = 0; trial < 20; trial++) {
    reset(60, 35, -Math.PI / 2); S.odor = { x: 100, y: 20 };
    for (let i = 0; i < 60; i++) step();
    const f = S.fly, a = f.h + (Math.random() - 0.5) * 2;
    S.threat = { x: f.x + Math.cos(a) * 8, y: f.y + Math.sin(a) * 8, r: 2, age: 0, loom: 0 };
    const e0 = escapes;
    for (let i = 0; i < 120; i++) { step(); if (escapes > e0) { hit++; lat.push(S.t); break; } }
  }
  out.threat = { escaped: hit + '/20' };
  // 4) idle: nothing in the arena for 20 s -> no escapes, fly keeps moving
  reset(60, 35, -Math.PI / 2); let moved = 0;
  for (let i = 0; i < 60 * 20; i++) { step(); moved += S.fly.spd * dt; }
  out.idle = { escapes, distanceFlown: +moved.toFixed(0) };
  out.act = Object.fromEntries(Object.entries(A).map(([k, v]) => [k, +v.toFixed(2)]));
  return out;
}, [overrides, bias, process.env.SCALE_KINDS || '{}']);
console.log(JSON.stringify(res, null, 1));
await browser.close();
})();
