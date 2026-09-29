// Speed and firing rates of the in-browser brain, run in Node (same JavaScript engine as Chrome).
// Usage: node brain/bench.cjs [windows per stimulus, default 20]
const fs = require('fs'), zlib = require('zlib'), path = require('path');
const { decode, FullBrain } = require('./fullbrain.js');

let t = Date.now();
const raw = zlib.gunzipSync(fs.readFileSync(path.join(__dirname, 'flywire783.bin.gz')));
const data = decode(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
const brain = new FullBrain(data);
console.log(`loaded ${data.n} neurons, ${data.m} connections in ${Date.now() - t} ms`);

const N = +process.argv[2] || 20;
const stimuli = {
  'nothing': {},
  'odor left': { antL: 1, antR: 0.3 },
  'light left': { eyeL: 1, eyeR: 0.2 },
  'looming': { loom: 1 },
};
for (const [name, s] of Object.entries(stimuli)) {
  const acc = {}; let wall = 0;
  for (let k = 0; k < N; k++) {
    const o = brain.window(s, 50);
    wall += o.wallMs;
    for (const g in o.rates) acc[g] = (acc[g] || 0) + o.rates[g] / N;
  }
  const x = wall / N / 50;
  console.log(`${name.padEnd(11)} ${(wall / N).toFixed(0).padStart(4)} ms per 50 ms of brain ` +
    `(${x <= 1 ? 'faster than real time' : x.toFixed(1) + 'x slower than real time'})  ` +
    ['al', 'mb', 'gf', 'dnL', 'dnR'].map(g => `${g} ${acc[g].toFixed(1)} Hz`).join('  '));
}
