/* Web Worker that runs the whole FlyWire brain next to the page (brain/fullbrain.js).
 * Speaks the same messages as brain_server/server.py, so the page does not care where the
 * brain runs:
 *   page -> worker: {t: 'sense', eyeL, eyeR, antL, antR, loom} (0..1), {t: 'stop'}
 *   worker -> page: {t: 'load', loaded, total, cached}, {t: 'status', msg, ready},
 *                   {t: 'brain', rates, top, simMs, wallMs}, {t: 'error', msg}
 * The data file (~21 MB) is downloaded once and kept in the browser's Cache Storage.
 */
'use strict';
importScripts('fullbrain.js');

const DATA_URL = new URL('flywire783.bin.gz', self.location.href).href;
const CACHE = 'zvuv-brain-v1';   // bump when the data file changes
const WINDOW_MS = 50;            // simulated time per step, as in brain_server/server.py
const MAX_RATE = 150;            // Hz for a sensor value of 1 (Shiu et al. 2024 default)
let sensors = {}, running = true;

onmessage = e => {
  const m = e.data || {};
  if (m.t === 'sense') sensors = m;
  if (m.t === 'stop') { running = false; close(); }
};

async function download() {
  let cache = null, res = null;
  try { cache = await caches.open(CACHE); res = await cache.match(DATA_URL); } catch (e) { /* no Cache Storage (private mode, file://) */ }
  if (res) {
    const buf = await res.arrayBuffer();
    postMessage({ t: 'load', loaded: buf.byteLength, total: buf.byteLength, cached: true });
    return buf;
  }
  res = await fetch(DATA_URL);
  if (!res.ok) throw new Error(`הורדת נתוני המוח נכשלה (${res.status})`);
  const total = +res.headers.get('Content-Length') || 21.3e6;
  const reader = res.body.getReader(), chunks = [];
  let loaded = 0, lastPost = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value); loaded += value.length;
    if (performance.now() - lastPost > 150) { lastPost = performance.now(); postMessage({ t: 'load', loaded, total, cached: false }); }
  }
  const blob = new Blob(chunks);
  postMessage({ t: 'load', loaded, total: loaded, cached: false });
  try { if (cache) await cache.put(DATA_URL, new Response(blob, { headers: { 'Content-Type': 'application/octet-stream' } })); } catch (e) { /* quota: works, just not stored */ }
  return blob.arrayBuffer();
}

async function unpack(buf) {
  const head = new Uint8Array(buf, 0, 2);
  if (head[0] !== 0x1f || head[1] !== 0x8b) return buf; // the server already un-gzipped it
  const stream = new Blob([buf]).stream().pipeThrough(new DecompressionStream('gzip'));
  return new Response(stream).arrayBuffer();
}

const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  try {
    const t0 = performance.now();
    const raw = await download();
    postMessage({ t: 'status', msg: 'פותח את נתוני המוח…', ready: false });
    const brain = new ZvuvBrain.FullBrain(ZvuvBrain.decode(await unpack(raw)));
    postMessage({ t: 'status', ready: true,
      msg: `המוח המלא נטען: ${brain.n.toLocaleString('he-IL')} נוירונים ו־${(brain.m / 1e6).toFixed(1)} מיליון חיבורים (${((performance.now() - t0) / 1000).toFixed(1)} שניות).` });
    while (running) {
      const start = performance.now();
      const out = brain.window(sensors, WINDOW_MS, MAX_RATE);
      delete out.synEvents;
      postMessage(out);
      // never run ahead of real time; when slower than real time, just yield for incoming messages
      await sleep(Math.max(0, WINDOW_MS - (performance.now() - start)));
    }
  } catch (err) {
    postMessage({ t: 'error', msg: String(err && err.message || err) });
  }
})();
