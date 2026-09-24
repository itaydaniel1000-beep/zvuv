#!/usr/bin/env python3
"""Does the full FlyWire brain model steer toward an odor?

The connectome is not touched. We vary only
  * the input: which olfactory receptor neurons (ORNs) get the odor, and how strongly, and
  * the readout: which descending neurons (DNs) we look at,
and ask whether any left/right DN pair responds more to an odor on its own side.

Each trial = 1 s from rest (as in Shiu et al. 2024), same equations and constants as server.py.
Run:  python brain_server/odor_probe.py           strong odor (paper-style 50/150 Hz), all DNs
      python brain_server/odor_probe.py --sweep   weak odor, 5-100 Hz: where does side information get lost?
      python brain_server/odor_probe.py --timecourse   10 ms bins after odor onset (antennal lobe ignition)
      python brain_server/odor_probe.py --pn      drive fruit-glomerulus PNs of one side directly
(each about 5-15 min with a C++ compiler, longer without). Writes data/odor_probe.json / odor_sweep.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import server  # noqa: E402
from brian2 import defaultclock, ms  # noqa: E402

FRUIT = ['ORN_DM1', 'ORN_DM2', 'ORN_DM3', 'ORN_DM4', 'ORN_VA2', 'ORN_VM2']
INPUTS = {
    'fruit esters (DM1,DM2,DM3,DM4,VA2,VM2)': FRUIT,
    'DM1 only (Or42b, vinegar attraction)': ['ORN_DM1'],
    'all ORNs': 'all',
}
STIMULI = {  # (antL, antR)
    'left only': (1.0, 0.0), 'right only': (0.0, 1.0),
    'left > right': (1.0, 0.5), 'right > left': (0.5, 1.0),
}
RATES = [150, 50]      # Hz for a sensor value of 1 (150 = paper default)
TRIAL_MS = 1000
REPEATS = 2


def main():
    defaultclock.dt = 0.1 * ms
    ann = pd.read_csv(server.bb.RAW / 'Supplemental_file1_neuron_annotations.tsv', sep='\t', dtype=str)
    comp = pd.read_csv(server.bb.RAW / 'Completeness_783.csv', index_col=0)
    ann = ann.drop_duplicates('root_id').set_index('root_id').reindex(comp.index.astype(str)).fillna('')
    dn = np.flatnonzero((ann['super_class'] == 'descending').values)
    dn_type, dn_side = ann['cell_type'].values[dn], ann['side'].values[dn]

    results = []
    for in_name, types in INPUTS.items():
        itypes = dict(server.INPUT_TYPES, antL=types, antR=types)
        for rate in RATES:
            brain = server.Brain(rate, 'auto', memory=False, input_types=itypes)
            n_in = len(brain.drive_idx['antL']), len(brain.drive_idx['antR'])
            for st_name, (l, r) in STIMULI.items():
                hz = np.zeros(len(dn))
                al = 0.0
                t0 = time.time()
                for _ in range(REPEATS):
                    out = brain.step(dict(antL=l, antR=r), TRIAL_MS)
                    hz += brain.hz[dn] / REPEATS
                    al += out['rates']['al'] / REPEATS
                res = dict(input=in_name, rate=rate, stim=st_name, n_orn=n_in, al=round(al, 1),
                           dn={f'{t}_{s[:1].upper()}#{i}': round(float(h), 1)
                               for i, (t, s, h) in enumerate(zip(dn_type, dn_side, hz)) if h > 0})
                results.append(res)
                a01 = {s: hz[(dn_type == 'DNa01') & (dn_side == s)].mean() for s in ('left', 'right')}
                a02 = {s: hz[(dn_type == 'DNa02') & (dn_side == s)].mean() for s in ('left', 'right')}
                print(f'{in_name[:22]:22} {rate:3d}Hz {st_name:12} PN {al:5.1f}  '
                      f'DNa01 L/R {a01["left"]:5.1f}/{a01["right"]:5.1f}  DNa02 L/R {a02["left"]:5.1f}/{a02["right"]:5.1f}  '
                      f'active DNs {int((hz > 0).sum())}  ({time.time() - t0:.0f}s)', flush=True)
            del brain

    out_path = server.ROOT / 'data' / 'odor_probe.json'
    out_path.write_text(json.dumps(results, indent=1))
    summarize(results, dn_type, dn_side)


def summarize(results, dn_type, dn_side):
    """Steering index per DN type: how much more the left member fires than the right one
    for odor on the left, compared with odor on the right. Positive = ipsilateral (turns toward
    the odor if the DN steers ipsilaterally, like DNa02)."""
    print('\nDN types with a left/right pair, steering index (Hz) = [(L-R) odor-left - (L-R) odor-right] / 2')

    def lr(res, t):
        vals = {'L': [], 'R': []}
        for k, v in res['dn'].items():
            name, side = k.split('#')[0].rsplit('_', 1)
            if name == t:
                vals[side].append(v)
        return (np.mean(vals['L']) if vals['L'] else 0.0), (np.mean(vals['R']) if vals['R'] else 0.0)

    paired = sorted({t for t, s in zip(dn_type, dn_side) if t and s == 'left'} &
                    {t for t, s in zip(dn_type, dn_side) if t and s == 'right'})
    for (in_name, rate) in sorted({(r['input'], r['rate']) for r in results}, key=lambda x: (x[0], -x[1])):
        by = {r['stim']: r for r in results if r['input'] == in_name and r['rate'] == rate}
        rows = []
        for t in paired:
            (lL, lR), (rL, rR) = lr(by['left only'], t), lr(by['right only'], t)
            (gL, gR), (hL, hR) = lr(by['left > right'], t), lr(by['right > left'], t)
            si = ((lL - lR) - (rL - rR)) / 2
            si2 = ((gL - gR) - (hL - hR)) / 2
            if max(lL, lR, rL, rR) > 0:
                rows.append((si, si2, t, lL, lR, rL, rR))
        rows.sort(key=lambda x: -abs(x[0]))
        print(f'\n{in_name}, {rate} Hz  ({len(rows)} DN pairs respond)')
        for si, si2, t, lL, lR, rL, rR in rows[:10]:
            print(f'  {t:10} SI {si:+6.1f} (graded {si2:+6.1f})   odor-left L/R {lL:5.1f}/{lR:5.1f}   odor-right L/R {rL:5.1f}/{rR:5.1f}')


def sweep():
    """Weak inputs: find the regime where the antennal lobe does not saturate, and check
    whether left/right information survives to the PNs and to the DNs there."""
    defaultclock.dt = 0.1 * ms
    ann = pd.read_csv(server.bb.RAW / 'Supplemental_file1_neuron_annotations.tsv', sep='\t', dtype=str)
    comp = pd.read_csv(server.bb.RAW / 'Completeness_783.csv', index_col=0)
    ann = ann.drop_duplicates('root_id').set_index('root_id').reindex(comp.index.astype(str)).fillna('')
    ct, side, cc, sc = (ann[c].values for c in ('cell_type', 'side', 'cell_class', 'super_class'))
    pn = {s: np.flatnonzero((cc == 'ALPN') & (side == s)) for s in ('left', 'right')}
    ln = np.flatnonzero(cc == 'ALLN')
    dn = np.flatnonzero(sc == 'descending')
    rows = []
    for in_name, types in INPUTS.items():
        itypes = dict(server.INPUT_TYPES, antL=types, antR=types)
        brain = server.Brain(1.0, 'auto', memory=False, input_types=itypes)
        for rate in SWEEP_RATES:
            brain.max_rate = rate
            r = {}
            for st, (l, rr) in (('left only', (1, 0)), ('right only', (0, 1))):
                acc = dict(pnL=0, pnR=0, ln=0, dns=0, dn=np.zeros(len(dn)))
                for _ in range(REPEATS):
                    brain.step(dict(antL=l, antR=rr), SWEEP_MS)
                    h = brain.hz
                    acc['pnL'] += h[pn['left']].mean() / REPEATS
                    acc['pnR'] += h[pn['right']].mean() / REPEATS
                    acc['ln'] += h[ln].mean() / REPEATS
                    acc['dn'] += h[dn] / REPEATS
                acc['dns'] = int((acc['dn'] > 0).sum())
                r[st] = acc
            # steering index for DNa02 / DNa01 and the best-lateralized DN pair
            def pair(t, st):
                return [r[st]['dn'][(ct[dn] == t) & (side[dn] == s)].mean() for s in ('left', 'right')]
            si = {}
            for t in sorted(set(ct[dn][side[dn] == 'left']) & set(ct[dn][side[dn] == 'right']) - {''}):
                (a, b), (c, d) = pair(t, 'left only'), pair(t, 'right only')
                if max(a, b, c, d) > 0:
                    si[t] = ((a - b) - (c - d)) / 2
            best = sorted(si.items(), key=lambda kv: -abs(kv[1]))[:4]
            L, R = r['left only'], r['right only']
            row = dict(input=in_name, rate=rate, pn_left_odor=[round(L['pnL'], 1), round(L['pnR'], 1)],
                       pn_right_odor=[round(R['pnL'], 1), round(R['pnR'], 1)], ln=round(L['ln'], 1),
                       active_dns=[L['dns'], R['dns']], dna02=[pair('DNa02', 'left only'), pair('DNa02', 'right only')],
                       si={k: round(v, 1) for k, v in best})
            rows.append(row)
            print(f"{in_name[:14]:14} {rate:5.0f}Hz  PN L/R: odor-left {L['pnL']:5.1f}/{L['pnR']:5.1f}  odor-right {R['pnL']:5.1f}/{R['pnR']:5.1f}"
                  f"  LN {L['ln']:5.1f}  DNs {L['dns']:3d}/{R['dns']:3d}  DNa02 L/R {pair('DNa02', 'left only')[0]:4.1f}/{pair('DNa02', 'left only')[1]:4.1f}"
                  f" vs {pair('DNa02', 'right only')[0]:4.1f}/{pair('DNa02', 'right only')[1]:4.1f}  top SI " +
                  ', '.join(f'{k} {v:+.1f}' for k, v in best), flush=True)
        del brain
    (server.ROOT / 'data' / 'odor_sweep.json').write_text(json.dumps(rows, indent=1, default=float))


SWEEP_RATES = [5, 10, 20, 30, 40, 60, 100]
SWEEP_MS = 500

FRUIT_PNS = ['DM1_lPN', 'DM2_lPN', 'DM3_adPN', 'DM3_vPN', 'DM4_adPN', 'DM4_vPN', 'VA2_adPN', 'VM2_adPN']


def _ann():
    ann = pd.read_csv(server.bb.RAW / 'Supplemental_file1_neuron_annotations.tsv', sep='\t', dtype=str)
    comp = pd.read_csv(server.bb.RAW / 'Completeness_783.csv', index_col=0)
    ann = ann.drop_duplicates('root_id').set_index('root_id').reindex(comp.index.astype(str)).fillna('')
    return (ann[c].values for c in ('cell_type', 'side', 'cell_class', 'super_class'))


def timecourse():
    """10 ms bins after odor onset: when does the antennal lobe ignite, and which side leads?"""
    defaultclock.dt = 0.1 * ms
    ct, side, cc, _ = _ann()
    pn = {s: np.flatnonzero((cc == 'ALPN') & (side == s)) for s in ('left', 'right')}
    ln = np.flatnonzero(cc == 'ALLN')
    a02 = {s: np.flatnonzero((ct == 'DNa02') & (side == s)) for s in ('left', 'right')}
    b = server.Brain(1, 'auto', memory=True)
    for rate in [1, 2, 5]:
        b.max_rate = rate
        for st, (l, r) in (('left', (1, 0)), ('right', (0, 1))):
            b.net.restore('rest')
            b.last_count[:] = 0
            bins = []
            for _ in range(30):
                b.step(dict(antL=l, antR=r), 10)
                h = b.hz
                bins.append(f"{h[pn['left']].mean():.0f}/{h[pn['right']].mean():.0f}|{h[ln].mean():.0f}|"
                            f"{h[a02['left']].mean():.0f}/{h[a02['right']].mean():.0f}")
            print(f'{rate} Hz fruit ORNs, odor {st}; 10 ms bins 0-300 ms (PN L/R | LN | DNa02 L/R):')
            print('  ' + '  '.join(bins[:15]) + '\n  ' + '  '.join(bins[15:]), flush=True)


def pn_drive():
    """Skip the ORNs: drive the fruit-glomerulus uPNs of one hemisphere directly."""
    defaultclock.dt = 0.1 * ms
    ct, side, cc, sc = _ann()
    server.bb.GROUPS['antL'] = dict(cell_type=FRUIT_PNS, side='left', label='')
    server.bb.GROUPS['antR'] = dict(cell_type=FRUIT_PNS, side='right', label='')
    b = server.Brain(1, 'auto', memory=False, input_types=dict(server.INPUT_TYPES, antL='all', antR='all'))
    driven = np.concatenate([b.drive_idx['antL'], b.drive_idx['antR']])
    other = {s: np.setdiff1d(np.flatnonzero((cc == 'ALPN') & (side == s)), driven) for s in ('left', 'right')}
    dn = np.flatnonzero(sc == 'descending')
    for rate in [5, 20, 50, 100]:
        b.max_rate = rate
        for st, (l, r) in (('left', (1, 0)), ('right', (0, 1))):
            h = np.mean([b.step(dict(antL=l, antR=r), 500) and b.hz.copy() for _ in range(REPEATS)], axis=0)
            a02 = [h[dn][(ct[dn] == 'DNa02') & (side[dn] == s)].mean() for s in ('left', 'right')]
            print(f"{rate:3d} Hz, {len(b.drive_idx['antL'])} uPNs {st}: other PNs L/R {h[other['left']].mean():5.1f}/"
                  f"{h[other['right']].mean():5.1f}  active DNs {(h[dn] > 0).sum():3d}  DNa02 L/R {a02[0]:4.1f}/{a02[1]:4.1f}", flush=True)


if __name__ == '__main__':
    mode = next((a for a in sys.argv[1:] if a.startswith('--')), '')
    dict([('--sweep', sweep), ('--timecourse', timecourse), ('--pn', pn_drive)]).get(mode, main)()
