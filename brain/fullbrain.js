/* The whole FlyWire fly brain (138,639 neurons, 15M connections) in JavaScript.
 *
 * Same model as Shiu et al. 2024 (model.py) and brain_server/server.py:
 *   dv/dt = (v_0 - v + g) / t_mbr,  dg/dt = -g / tau   (both frozen while refractory)
 *   spike when v > v_th -> v = v_rst, g = 0, refractory 2.2 ms; synaptic input that arrives
 *   while a neuron is refractory is lost (that is what Brian2 does with "(unless refractory)")
 *   each spike adds (synapse count x sign x w_syn) to g of every target, 1.8 ms later
 *   sensory drive: Poisson spikes that add w_syn x f_poi to v of the driven neurons
 * integrated exactly (the equations are linear) with dt = 0.1 ms.
 *
 * Speed trick (exact, not an approximation): a neuron is only stepped while it could reach
 * threshold. Without new input its future is known in closed form, and its highest future v is
 * max(0, v - v_0) + max(0, g) x K. When that is below threshold the neuron "sleeps"; when an
 * input arrives it jumps straight to the current step with the exact solution. Most of the brain
 * sleeps most of the time.
 *
 * Data: brain/flywire783.bin.gz, built by data/build_web_brain.py.
 */
(function (root) {
  'use strict';

  function decode(buf) {
    const u8 = new Uint8Array(buf);
    const magic = String.fromCharCode(...u8.subarray(0, 8));
    if (magic !== 'ZVUVBRN1') throw new Error('not a zvuv brain file');
    const hlen = new DataView(buf).getUint32(8, true);
    const header = JSON.parse(new TextDecoder().decode(u8.subarray(12, 12 + hlen)));
    let off = 12 + hlen; off += (4 - off % 4) % 4;
    const { n, m } = header;
    const take = (Type, len) => { const a = new Type(buf, off, len); off += len * Type.BYTES_PER_ELEMENT; return a; };
    const outdeg = take(Uint32Array, n);
    const type = take(Uint16Array, n);
    const sign = take(Int8Array, n);
    const side = take(Uint8Array, n);
    const group = take(Uint8Array, n);
    const drive = take(Uint8Array, n);
    const d0 = take(Uint8Array, m), d1 = take(Uint8Array, m), d2 = take(Uint8Array, m);
    const c0 = take(Uint8Array, m), c1 = take(Uint8Array, m);
    // CSR: rowStart[i]..rowStart[i+1] are neuron i's connections
    const rowStart = new Uint32Array(n + 1);
    for (let i = 0; i < n; i++) rowStart[i + 1] = rowStart[i] + outdeg[i];
    const target = new Uint32Array(m);
    const weight = new Float32Array(m); // mV added to g of the target per presynaptic spike
    const w = header.params.w_syn;
    for (let i = 0; i < n; i++) {
      let prev = 0;
      const s = sign[i] * w;
      for (let k = rowStart[i], e = rowStart[i + 1], first = true; k < e; k++) {
        const dlt = d0[k] | (d1[k] << 8) | (d2[k] << 16);
        prev = first ? dlt : prev + dlt; first = false;
        target[k] = prev;
        weight[k] = s * (1 + (c0[k] | (c1[k] << 8)));
      }
    }
    return { header, n, m, rowStart, target, weight, type: type.slice(), side: side.slice(), group: group.slice(), drive: drive.slice() };
  }

  class FullBrain {
    constructor(data, rng = Math.random) {
      Object.assign(this, data);
      const p = data.header.params;
      this.p = p;
      this.rng = rng;
      this.dt = p.dt;
      this.a = Math.exp(-p.dt / p.t_mbr);                          // v decay per step
      this.b = Math.exp(-p.dt / p.tau);                            // g decay per step
      this.c = p.tau / (p.tau - p.t_mbr) * (this.b - this.a);      // g -> v per step (exact)
      this.th = p.v_th - p.v_0;                                    // threshold above rest
      this.rst = p.v_rst - p.v_0;
      this.kick = p.w_syn * p.f_poi;                               // one Poisson input spike
      this.refSteps = Math.round(p.t_rfc / p.dt);
      this.delaySteps = Math.round(p.t_dly / p.dt);
      const n = this.n;
      // Per-neuron state, interleaved so one synaptic event touches one cache line:
      //   S[4i] = v - v_0 (mV), S[4i+1] = g (mV), I[4i+2] = refractory until this step,
      //   I[4i+3] = step a sleeping neuron's state refers to, or -1 while it is awake
      this.S = new Float32Array(4 * n);
      this.I = new Int32Array(this.S.buffer);
      this.active = new Int32Array(n);
      this.nActive = 0;
      // exact jumps over n steps without input: x' = x A[n] + g C[n], g' = g B[n]
      const T = 20000, k = p.tau / (p.tau - p.t_mbr);
      this.An = new Float64Array(T); this.Bn = new Float64Array(T); this.Cn = new Float64Array(T);
      this.An[0] = this.Bn[0] = 1; this.K = 0;
      for (let s = 1; s < T; s++) {
        this.An[s] = this.An[s - 1] * this.a; this.Bn[s] = this.Bn[s - 1] * this.b;
        this.Cn[s] = k * (this.Bn[s] - this.An[s]);
        if (this.Cn[s] > this.K) this.K = this.Cn[s];   // largest possible rise of v per mV of g
      }
      this.count = new Uint32Array(n);     // spikes in the current window
      // spikes waiting for their 1.8 ms synaptic delay
      this.ring = Array.from({ length: this.delaySteps + 1 }, () => new Int32Array(n));
      this.ringLen = new Int32Array(this.delaySteps + 1);
      this.spk = new Int32Array(n);
      this.step = 0;
      this.synEvents = 0;
      // driven neurons per sensor (no refractory period, as in model.py)
      this.driven = this.header.sensors.map((_, si) => {
        const idx = [];
        for (let i = 0; i < n; i++) if (this.drive[i] === si + 1) idx.push(i);
        return Int32Array.from(idx);
      });
      this.noRef = new Uint8Array(n);
      for (const d of this.driven) for (const i of d) this.noRef[i] = 1;
      this.members = this.header.groups.map((_, gi) => {
        const idx = [];
        for (let i = 0; i < n; i++) if (this.group[i] === gi + 1) idx.push(i);
        return Int32Array.from(idx);
      });
      this.rates = new Float64Array(this.driven.length);
    }

    /** Back to rest: no activity, nothing in flight (each window starts like a trial in the paper). */
    reset() {
      this.S.fill(0);
      this.nActive = 0; this.ringLen.fill(0); this.count.fill(0); this.step = 0; this.synEvents = 0;
    }

    /** sensors: {eyeL, eyeR, antL, antR, loom} in 0..1, maxRate: Hz for a value of 1 */
    setInput(sensors, maxRate) {
      this.header.sensors.forEach((s, i) => { this.rates[i] = Math.max(0, Math.min(1, +sensors[s] || 0)) * maxRate; });
    }

    /** Add dv (mV) to v of neuron i, as if it happened after the integration of step t. */
    inject(i, dv, t = this.step - 1) { this.wake(i, t); this.S[4 * i] += dv; }

    /** Bring a sleeping neuron up to step t in one exact jump, then keep it updated every step. */
    wake(i, t) {
      const S = this.S, I = this.I, o = 4 * i, last = I[o + 3];
      if (last < 0) return;
      const n = t - last;
      if (n > 0 && (S[o] !== 0 || S[o + 1] !== 0)) {
        if (n < this.An.length) { S[o] = S[o] * this.An[n] + S[o + 1] * this.Cn[n]; S[o + 1] *= this.Bn[n]; }
        else { S[o] = 0; S[o + 1] = 0; }
      }
      I[o + 3] = -1; this.active[this.nActive++] = i;
    }

    run(steps) {
      const { S, I, active, rowStart, target, weight, noRef, count, An, Bn, Cn } = this;
      const a = this.a, b = this.b, c = this.c, th = this.th, rst = this.rst, K = this.K, T = An.length;
      const slots = this.ring.length;
      for (let s = 0; s < steps; s++) {
        const t = this.step;
        // 1) integrate awake neurons and 2) threshold. A neuron that cannot reach threshold
        //    without new input (max of its future trajectory < v_th) goes to sleep: skipping it is exact.
        let nSpk = 0, keep = 0;
        const spk = this.spk;
        for (let k = 0; k < this.nActive; k++) {
          const i = active[k], o = 4 * i;
          let xi = S[o], gi = S[o + 1];
          if (t >= I[o + 2]) {
            xi = xi * a + gi * c; gi *= b;
            S[o] = xi; S[o + 1] = gi;
            if (xi > th) spk[nSpk++] = i;
          }
          if ((xi > 0 ? xi : 0) + (gi > 0 ? gi * K : 0) < th) I[o + 3] = t;
          else active[keep++] = i;
        }
        this.nActive = keep;
        // 3a) Poisson sensory drive (geometric skipping: cost ~ number of input spikes)
        for (let si = 0; si < this.driven.length; si++) {
          const p = this.rates[si] * this.dt * 1e-3;
          if (p <= 0) continue;
          const d = this.driven[si], L = Math.log(1 - p);
          for (let j = Math.floor(Math.log(1 - this.rng()) / L); j < d.length; j += 1 + Math.floor(Math.log(1 - this.rng()) / L)) {
            this.inject(d[j], this.kick, t);
          }
        }
        // 3b) deliver spikes fired delaySteps ago
        const due = t % slots, dueList = this.ring[due], dueLen = this.ringLen[due];
        let nAct = this.nActive;
        for (let k = 0; k < dueLen; k++) {
          const i = dueList[k];
          const e = rowStart[i + 1];
          for (let q = rowStart[i]; q < e; q++) {
            const j = target[q], o = 4 * j;
            if (t < I[o + 2]) continue; // Brian2 drops input that arrives while the target is refractory
            const last = I[o + 3];
            if (last >= 0) { // asleep: exact jump to now, then wake up
              const n = t - last;
              if (n > 0 && (S[o] !== 0 || S[o + 1] !== 0)) {
                if (n < T) { S[o] = S[o] * An[n] + S[o + 1] * Cn[n]; S[o + 1] *= Bn[n]; } else { S[o] = 0; S[o + 1] = 0; }
              }
              I[o + 3] = -1; active[nAct++] = j;
            }
            S[o + 1] += weight[q];
          }
          this.synEvents += e - rowStart[i];
        }
        this.nActive = nAct;
        this.ringLen[due] = 0;
        // 4) reset spiking neurons, queue their spikes for delivery after the delay
        const slot = (t + this.delaySteps) % slots, list = this.ring[slot];
        let len = this.ringLen[slot];
        for (let k = 0; k < nSpk; k++) {
          const i = spk[k], o = 4 * i;
          S[o] = rst; S[o + 1] = 0;
          if (!noRef[i]) I[o + 2] = t + this.refSteps; // Brian: refractory while (t - lastspike) < t_rfc
          count[i]++;
          list[len++] = i;
        }
        this.ringLen[slot] = len;
        this.step++;
      }
    }

    /** Run one window from rest and report like brain_server/server.py does. */
    window(sensors, ms, maxRate = 150) {
      this.reset();
      this.setInput(sensors, maxRate);
      const t0 = now();
      this.run(Math.round(ms / this.dt));
      const wall = now() - t0;
      const hz = i => this.count[i] / (ms / 1000);
      const rates = {};
      this.header.groups.forEach((gname, gi) => {
        const mem = this.members[gi]; let s = 0;
        for (const i of mem) s += this.count[i];
        rates[gname] = Math.round(s / Math.max(1, mem.length) / (ms / 1000) * 100) / 100;
      });
      // most active neurons
      const best = [];
      for (let i = 0; i < this.n; i++) if (this.count[i]) best.push(i);
      best.sort((u, v) => this.count[v] - this.count[u]);
      const sides = ['', '_L', '_R', '_C'];
      const top = best.slice(0, 8).map(i => [(this.header.types[this.type[i]] || '?') + sides[this.side[i]], Math.round(hz(i) * 10) / 10]);
      return { t: 'brain', rates, top, simMs: ms, wallMs: Math.round(wall), synEvents: this.synEvents };
    }
  }

  const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
  const api = { decode, FullBrain };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ZvuvBrain = api;
})(typeof self !== 'undefined' ? self : this);
