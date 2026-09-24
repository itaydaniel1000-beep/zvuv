#!/usr/bin/env python3
"""The full FlyWire brain (Shiu et al. 2024) as a WebSocket server for the zvuv site.

The site sends its virtual (or robot) sensors; the server turns them into Poisson
input to the real sensory neurons, runs the spiking model for a short step and
sends back the firing rate of every node the site draws (DNa01/02, DNa03, Giant Fiber ...).

Model: leaky integrate-and-fire network with every proofread FlyWire v783 neuron
(~139k) and connection (~15M), same equations and constants as model.py in
github.com/philshiu/Drosophila_brain_model (MIT License, Philip Shiu and Nico Spiller).
Neuron groups: the same GROUPS as data/build_brain.py.

Run:  python brain_server/server.py            (see docs/brain-server-windows.md)
      python brain_server/server.py --selftest (no site needed: odor left vs right)
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'data'))
import build_brain as bb  # noqa: E402  (data download + neuron groups)

from brian2 import (NeuronGroup, Synapses, PoissonGroup, SpikeMonitor, Network,  # noqa: E402
                    mV, ms, Hz, prefs, defaultclock)

# ---- constants from Shiu et al. 2024, model.py ----------------------------------------
P = dict(
    v_0=-52 * mV, v_rst=-52 * mV, v_th=-45 * mV, t_mbr=20 * ms, tau=5 * ms,
    t_rfc=2.2 * ms, t_dly=1.8 * ms, w_syn=0.275 * mV, f_poi=250,
)
EQS = '''
dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
dg/dt = -g / tau               : volt (unless refractory)
rfc                            : second
'''

SENSORS = ('eyeL', 'eyeR', 'antL', 'antR', 'loom')   # what the site sends, each 0..1
# Which neurons of each sensory group get the Poisson drive. Driving all 5,000 photoreceptors
# or all 1,100 ORNs of a side at once sends the whole network into runaway excitation, so each
# sensor drives the cell types that carry that stimulus:
#   light -> R7 + R8 (UV / blue phototaxis runs through R7/R8), odor -> ORNs of the glomeruli
#   that respond to fruit / fermentation esters (DM1 = Or42b, DM2, DM3, DM4, VA2, VM2).
INPUT_TYPES = {
    'eyeL': ['R7', 'R8'], 'eyeR': ['R7', 'R8'],
    'antL': ['ORN_DM1', 'ORN_DM2', 'ORN_DM3', 'ORN_DM4', 'ORN_VA2', 'ORN_VM2'],
    'antR': ['ORN_DM1', 'ORN_DM2', 'ORN_DM3', 'ORN_DM4', 'ORN_VA2', 'ORN_VM2'],
    'loom': ['LPLC2', 'LC4'],
}
REPORT = ('eyeL', 'eyeR', 'antL', 'antR', 'loom', 'ol', 'al', 'mb', 'cx', 'gf', 'dnL', 'dnR', 'dnF')


class Brain:
    def __init__(self, max_rate, target, memory=False, input_types=None):
        prefs.codegen.target = target
        bb.fetch()
        comp = pd.read_csv(bb.RAW / 'Completeness_783.csv', index_col=0)
        con = pd.read_parquet(bb.RAW / 'Connectivity_783.parquet')
        ann = pd.read_csv(bb.RAW / 'Supplemental_file1_neuron_annotations.tsv', sep='\t', dtype=str)
        ann = ann.drop_duplicates('root_id').set_index('root_id').reindex(comp.index.astype(str)).fillna('')
        masks = bb.masks(ann)
        self.idx = {k: np.flatnonzero(m) for k, m in masks.items()}
        self.names = ann['cell_type'].values
        self.sides = ann['side'].values
        self.max_rate = max_rate
        self.memory = memory
        n = len(comp)

        t0 = time.time()
        neu = NeuronGroup(n, EQS, method='linear', threshold='v > v_th',
                          reset='v = v_rst; g = 0 * mV', refractory='rfc', namespace=P, name='neurons')
        neu.v = P['v_0']
        neu.g = 0 * mV
        neu.rfc = P['t_rfc']
        syn = Synapses(neu, neu, 'w : volt', on_pre='g += w', delay=P['t_dly'], name='synapses')
        syn.connect(i=con['Presynaptic_Index'].values, j=con['Postsynaptic_Index'].values)
        syn.w = con['Excitatory x Connectivity'].values * P['w_syn']
        del con

        # One Poisson source per sensory neuron (the paper drives neurons the same way).
        types = ann['cell_type'].values
        input_types = input_types or INPUT_TYPES
        drive_idx = {s: self.idx[s] if input_types.get(s) == 'all' else self.idx[s][np.isin(types[self.idx[s]], input_types[s])]
                     for s in SENSORS}
        self.drive_idx = drive_idx
        self.inp_targets = np.concatenate([drive_idx[s] for s in SENSORS])
        self.inp_slices, start = {}, 0
        for s in SENSORS:
            self.inp_slices[s] = slice(start, start + len(drive_idx[s]))
            start += len(drive_idx[s])
        self.poi = PoissonGroup(len(self.inp_targets), rates=0 * Hz, name='sensors')
        drive = Synapses(self.poi, neu, on_pre='v += w_syn * f_poi', namespace=P, name='drive')
        drive.connect(i=np.arange(len(self.inp_targets)), j=self.inp_targets)
        neu.rfc[self.inp_targets] = 0 * ms  # as in model.py: no refractory period for driven neurons

        self.neu = neu
        self.spk = SpikeMonitor(neu, record=False)  # counts only
        self.net = Network(neu, syn, self.poi, drive, self.spk)
        self.net.store('rest')
        self.last_count = np.zeros(n, dtype=np.int64)
        self.sim_ms = 0.0
        print(f'network ready: {n} neurons, {len(syn)} connections, '
              f'{len(self.inp_targets)} sensory inputs ({time.time() - t0:.0f} s)', flush=True)

    def step(self, sensors, step_ms):
        r_in = np.zeros(len(self.inp_targets))
        for s in SENSORS:
            r_in[self.inp_slices[s]] = float(np.clip(sensors.get(s, 0), 0, 1)) * self.max_rate
        if not self.memory:
            # Every step starts from rest, like a trial in the paper. With memory, odor input
            # leaves the antennal lobe local neurons and APL firing at ~300 Hz indefinitely.
            # restore() also empties the synaptic delay queues and zeroes the spike counts.
            self.net.restore('rest')
            self.last_count[:] = 0
        self.poi.rates = r_in * Hz
        t0 = time.time()
        self.net.run(step_ms * ms, namespace={})  # all constants live in the groups' own namespace
        wall = (time.time() - t0) * 1000
        count = np.asarray(self.spk.count[:])
        diff, self.last_count = count - self.last_count, count
        hz = diff / (step_ms / 1000)
        self.hz = hz  # per-neuron rates of the last step (used by odor_probe.py)
        self.sim_ms += step_ms
        groups = {k: round(float(hz[self.idx[k]].mean()), 2) for k in REPORT}
        active = np.argsort(hz)[::-1][:8]
        top = [[f'{self.names[i] or "?"}{"_" + self.sides[i][0].upper() if self.sides[i] else ""}', round(float(hz[i]), 1)]
               for i in active if hz[i] > 0]
        return dict(t='brain', rates=groups, top=top, simMs=step_ms, wallMs=round(wall), total=self.sim_ms)


# Pages allowed to connect (browsers send Origin; 'null' = index.html opened from disk).
# Anything else is refused so random sites cannot use a public server's CPU.
DEFAULT_ORIGINS = 'https://itaydaniel1000-beep.github.io,null,http://localhost,http://127.0.0.1'

STATUS_PAGE = '''<!doctype html><meta charset="utf-8"><title>zvuv brain</title>
<body style="font-family:system-ui;background:#070b10;color:#dbe7f3;max-width:640px;margin:40px auto;padding:0 16px;line-height:1.6" dir="rtl">
<h1>🪰 המוח המלא של זבוב</h1>
<p>השרת פועל: {status}. {clients} אתרים מחוברים עכשיו.</p>
<p>זה שרת WebSocket, ואין מה לראות בו ישירות. פותחים את
<a style="color:#3ee6a8" href="https://itaydaniel1000-beep.github.io/zvuv/">מרכז השליטה</a>
ולוחצים „חבר למוח האמיתי”.</p>
<p style="color:#7f93a8">FlyWire v783 · מודל של Shiu et al. 2024 · 138,639 נוירונים</p>'''


async def serve(args):
    from http import HTTPStatus
    from websockets.asyncio.server import serve as ws_serve
    from websockets.datastructures import Headers
    from websockets.http11 import Response

    state = dict(brain=None, sensors={}, status='בונה את הרשת (כמה דקות)…', clients=set())

    async def broadcast(msg):
        data = json.dumps(msg, ensure_ascii=False)
        for ws in list(state['clients']):
            try:
                await ws.send(data)
            except Exception:
                state['clients'].discard(ws)

    async def handler(ws):
        state['clients'].add(ws)
        print('site connected', flush=True)
        await ws.send(json.dumps(dict(t='status', ready=state['brain'] is not None, msg=state['status']), ensure_ascii=False))
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except ValueError:
                    continue
                if msg.get('t') == 'sense':
                    state['sensors'] = {k: msg.get(k, 0) for k in SENSORS}
        finally:
            state['clients'].discard(ws)
            print('site disconnected', flush=True)

    async def sim_loop():
        loop = asyncio.get_running_loop()
        state['brain'] = await loop.run_in_executor(None, Brain, args.max_rate, args.target, args.memory)
        state['status'] = 'המוח המלא פועל'
        await broadcast(dict(t='status', ready=True, msg=state['status']))
        while True:
            if not state['clients']:  # nobody listening: don't burn the CPU
                await asyncio.sleep(0.2)
                continue
            out = await loop.run_in_executor(None, state['brain'].step, dict(state['sensors']), args.step)
            await broadcast(out)
            r = out['rates']
            print(f"sim {out['total'] / 1000:7.2f}s  step {out['wallMs']:5d} ms  "
                  f"dnL {r['dnL']:5.1f}  dnR {r['dnR']:5.1f}  dnF {r['dnF']:5.1f}  gf {r['gf']:5.1f} Hz", flush=True)

    def http_page(connection, request):
        # Plain HTTP (the Space's page and health check) gets a status page; WebSocket goes through.
        if request.headers.get('Upgrade', '').lower() == 'websocket':
            return None
        body = STATUS_PAGE.format(status=state['status'], clients=len(state['clients'])).encode()
        return Response(HTTPStatus.OK, 'OK', Headers([('Content-Type', 'text/html; charset=utf-8'),
                                                     ('Content-Length', str(len(body)))]), body)

    origins = [o.strip() or None for o in args.origins.split(',')] + [None] if args.origins != '*' else None
    async with ws_serve(handler, args.host, args.port, max_size=2 ** 16, origins=origins, process_request=http_page):
        print(f'listening on ws://{args.host}:{args.port}  (Ctrl+C to stop)', flush=True)
        await sim_loop()


def selftest(args):
    b = Brain(args.max_rate, args.target, args.memory)
    tests = [('odor left', dict(antL=1, antR=0.3)), ('odor right', dict(antL=0.3, antR=1)),
             ('light left', dict(eyeL=1, eyeR=0.2)), ('light right', dict(eyeL=0.2, eyeR=1)), ('looming', dict(loom=1))]
    for name, s in [('nothing', {})] + [x for t in tests for x in (t, ('  after', {}))]:
        acc = {k: 0.0 for k in REPORT}
        n = 5
        for _ in range(n):
            out = b.step(s, args.step)
            for k in REPORT:
                acc[k] += out['rates'][k] / n
        print(f'{name:11} step {out["wallMs"]:5d} ms  ' + '  '.join(f'{k} {acc[k]:.1f}' for k in ('al', 'mb', 'cx', 'dnL', 'dnR', 'dnF', 'gf')), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--host', default='localhost', help='0.0.0.0 to accept connections from outside (cloud)')
    ap.add_argument('--origins', default=os.environ.get('ALLOWED_ORIGINS', DEFAULT_ORIGINS),
                    help="comma-separated web pages allowed to connect, or '*' for any")
    ap.add_argument('--port', type=int, default=8765)
    ap.add_argument('--step', type=float, default=50, help='simulated milliseconds per step (default 50)')
    ap.add_argument('--max-rate', type=float, default=150, help='Poisson rate (Hz) for a sensor value of 1; 150 as in the paper')
    ap.add_argument('--target', default='auto', choices=['auto', 'numpy', 'cython'],
                    help='Brian2 code generation: cython is ~6x faster but needs a C++ compiler; '
                         'auto uses cython when a compiler works, numpy otherwise')
    ap.add_argument('--memory', action='store_true',
                    help='keep the network state between steps (default: every step starts from rest)')
    ap.add_argument('--selftest', action='store_true', help='run a few fixed stimuli and print DN rates, no WebSocket')
    args = ap.parse_args()
    defaultclock.dt = 0.1 * ms
    try:
        selftest(args) if args.selftest else asyncio.run(serve(args))
    except KeyboardInterrupt:
        print('bye')
