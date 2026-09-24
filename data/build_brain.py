#!/usr/bin/env python3
"""Build the website's brain wiring (SYN) from the real FlyWire connectome.

Every circle in the site's brain diagram stands for a set of real FlyWire cell
types. This script counts the real synapses between those sets, takes the sign
from the predicted neurotransmitter, turns the counts into model weights and
writes them into index.html (between the FLYWIRE:BEGIN / FLYWIRE:END markers)
and into data/flywire_brain.json.

Data (downloaded into data/raw/ on first run, ~140 MB, not committed):
  * Connectivity_783.parquet, Completeness_783.csv — FlyWire v783 connectivity
    as packaged by Shiu et al. 2024 (Nature), github.com/philshiu/Drosophila_brain_model.
    One row per connected neuron pair: synapse count and +1/-1 sign
    (GABA and glutamate = inhibitory, everything else = excitatory),
    from the neurotransmitter predictions of Eckstein et al. 2024.
  * Supplemental_file1_neuron_annotations.tsv — cell types, classes and hemisphere
    for every neuron, Schlegel et al. 2024 (Nature), github.com/flyconnectome/flywire_annotations.
The same data are on Zenodo (doi:10.5281/zenodo.10676866) and codex.flywire.ai.

Run:  pip install pandas pyarrow scipy && python data/build_brain.py
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

HERE = Path(__file__).resolve().parent
RAW = HERE / 'raw'
SITE = HERE.parent / 'index.html'

SHIU = 'https://raw.githubusercontent.com/philshiu/Drosophila_brain_model/91bdd1e7dcf193f3e7ca5a8933497fcef63b7960/'
ANNO = 'https://raw.githubusercontent.com/flyconnectome/flywire_annotations/8587524c1748ce5ef2080822a2fc890fc03bf597/supplemental_files/'
FILES = {
    'Connectivity_783.parquet': SHIU + 'Connectivity_783.parquet',
    'Completeness_783.csv': SHIU + 'Completeness_783.csv',
    'Supplemental_file1_neuron_annotations.tsv': ANNO + 'Supplemental_file1_neuron_annotations.tsv',
}

# ---------------------------------------------------------------- model knobs
# weight = sign * GAIN * |input fraction| ** EXPONENT
# "input fraction" = share of the target neurons' incoming synapses that come from the source
# (averaged over the target neurons). The cube root squeezes 4 orders of magnitude into a
# range a 16-node rate model can use; GAIN sets the overall scale.
GAIN = 7.0
EXPONENT = 1 / 3
MIN_FRAC = 1e-3   # a direct connection counts if it supplies >= 0.1% of the target's input
MIN_SYN = 100     # ... and has at least this many synapses in total
MIN_EFF = 1e-5    # smallest multi-synapse effective connectivity worth keeping
MAX_HOPS = 3

# ---------------------------------------------------------------- node -> cell types
# key: model node id. The other fields pick the FlyWire neurons; label is shown on the site (Hebrew).
GROUPS = {
    'eyeL': dict(cell_type=['R1-6', 'R7', 'R8'], side='left',
                 label='פוטורצפטורים R1–R6, R7, R8 בעין השמאלית'),
    'eyeR': dict(cell_type=['R1-6', 'R7', 'R8'], side='right',
                 label='פוטורצפטורים R1–R6, R7, R8 בעין הימנית'),
    'antL': dict(cell_class=['olfactory'], side='left',
                 label='נוירוני קולטני ריח (ORN) באנטנה השמאלית'),
    'antR': dict(cell_class=['olfactory'], side='right',
                 label='נוירוני קולטני ריח (ORN) באנטנה הימנית'),
    'loom': dict(cell_type=['LPLC2', 'LC4'],
                 label='LPLC2 + LC4: גלאי התקרבות (looming) שמזינים את Giant Fiber'),
    'ol':   dict(super_class=['optic', 'visual_projection'], exclude_type=['LPLC2', 'LC4'],
                 label='כל האונה האופטית (למינה, מדולה, לובולה, לובולה פלייט) ונוירוני ההקרנה החזותיים, חוץ מ-LPLC2 ו-LC4'),
    'al':   dict(cell_class=['ALPN'],
                 label='נוירוני ההקרנה של אונת הריח (ALPN)'),
    'mb':   dict(cell_class=['Kenyon_Cell', 'MBON'],
                 label='תאי קניון (Kenyon cells) ונוירוני הפלט של הגוף הפטרייתי (MBON)'),
    'cx':   dict(cell_class=['CX'],
                 label='כל תאי הקומפלקס המרכזי: EPG, PFL, PFN, נוירוני הטבעת ועוד'),
    'gf':   dict(cell_type=['DNp01'],
                 label='DNp01, ה-Giant Fiber, אחד בכל צד'),
    'dnL':  dict(cell_type=['DNa01', 'DNa02'], side='left',
                 label='DNa01 + DNa02 בצד שמאל: נוירוני היגוי'),
    'dnR':  dict(cell_type=['DNa01', 'DNa02'], side='right',
                 label='DNa01 + DNa02 בצד ימין: נוירוני היגוי'),
    'dnF':  dict(cell_type=['DNa03', 'DNp09'],
                 label='DNa03 (היעד העיקרי של הקומפלקס המרכזי) + DNp09 (P9, הליכה קדימה)'),
}
LAYER = dict(eyeL=0, eyeR=0, antL=0, antR=0, loom=0, ol=1, al=1, mb=2, cx=2, gf=2, dnL=3, dnF=3, dnR=3)
MIRROR = dict(eyeL='eyeR', eyeR='eyeL', antL='antR', antR='antL', dnL='dnR', dnR='dnL')

# Connections the current model already had between brain nodes. If there is no direct
# connection in the data we look for a 2–3 synapse pathway instead of dropping them outright.
MODEL_EDGES = [('eyeL', 'ol'), ('eyeR', 'ol'), ('antL', 'al'), ('antR', 'al'), ('al', 'mb'), ('ol', 'cx'),
               ('mb', 'dnF'), ('cx', 'dnF'), ('loom', 'gf'), ('gf', 'dnF'), ('dnL', 'dnR'), ('dnR', 'dnL')]

# Olfactory steering: ORN -> DNa01/02 has no direct synapses, it runs through 3 synapses
# (ORN -> PN -> lateral horn / LAL -> DN). The data give the sign and relative strength of each
# pathway; the ipsilateral gain is a model parameter.
ODOR_STEER_GAIN = 3.2
ODOR_CONTRA_W = -3.0

# Pieces the brain connectome cannot provide. They stay hand-set:
#  * eye -> DN: no consistent photoreceptor -> DNa01/02 pathway within 3 synapses.
#  * everything into wings / legs: motor neurons live in the ventral nerve cord (VNC),
#    which FlyWire does not include (that is the MANC connectome).
HAND_SET = [
    ('eyeL', 'dnL', 2.4, 'phototaxis shortcut, no pathway within 3 synapses'),
    ('eyeR', 'dnR', 2.4, 'phototaxis shortcut, no pathway within 3 synapses'),
    ('eyeL', 'dnR', -2.2, 'phototaxis shortcut, no pathway within 3 synapses'),
    ('eyeR', 'dnL', -2.2, 'phototaxis shortcut, no pathway within 3 synapses'),
    ('gf', 'leg', 3.0, 'VNC (GF -> TTMn jump muscle), not in FlyWire'),
    ('gf', 'wL', 1.5, 'VNC, not in FlyWire'), ('gf', 'wR', 1.5, 'VNC, not in FlyWire'),
    ('dnL', 'wR', 1.4, 'VNC, not in FlyWire'), ('dnR', 'wL', 1.4, 'VNC, not in FlyWire'),
    ('dnF', 'wL', 1.0, 'VNC, not in FlyWire'), ('dnF', 'wR', 1.0, 'VNC, not in FlyWire'),
]


def fetch():
    RAW.mkdir(exist_ok=True)
    for name, url in FILES.items():
        dst = RAW / name
        if dst.exists():
            continue
        print(f'downloading {name} ...', file=sys.stderr)
        tmp = dst.with_suffix(dst.suffix + '.part')
        urllib.request.urlretrieve(url, tmp)
        tmp.rename(dst)


def load():
    ids = pd.read_csv(RAW / 'Completeness_783.csv').iloc[:, 0].astype(str).values
    e = pd.read_parquet(RAW / 'Connectivity_783.parquet')
    a = pd.read_csv(RAW / 'Supplemental_file1_neuron_annotations.tsv', sep='\t', dtype=str)
    a = a.drop_duplicates('root_id').set_index('root_id').reindex(ids).fillna('')
    n = len(ids)
    pre, post = e['Presynaptic_Index'].values, e['Postsynaptic_Index'].values
    syn = e['Connectivity'].values.astype(float)
    sign = e['Excitatory'].values.astype(float)
    total_in = np.bincount(post, weights=syn, minlength=n)
    total_in[total_in == 0] = 1
    C = sp.csr_matrix((syn, (pre, post)), shape=(n, n))                  # synapse counts
    Cs = sp.csr_matrix((sign * syn, (pre, post)), shape=(n, n))          # signed counts
    Wt = sp.csr_matrix((sign * syn / total_in[post], (post, pre)), shape=(n, n))  # signed input fraction, transposed
    return a, C, Cs, Wt


def masks(a):
    out = {}
    for k, g in GROUPS.items():
        m = np.ones(len(a), bool)
        for col in ('cell_type', 'cell_class', 'super_class'):
            if col in g:
                m &= a[col].isin(g[col]).values
        if 'side' in g:
            m &= (a['side'] == g['side']).values
        if 'exclude_type' in g:
            m &= ~a['cell_type'].isin(g['exclude_type']).values
        out[k] = m
    return out


def main():
    fetch()
    a, C, Cs, Wt = load()
    M = masks(a)

    def direct(p, q):
        sub = C[M[p]][:, M[q]]
        frac = (Wt[M[q]][:, M[p]].sum(axis=1).A1).mean()
        return int(sub.sum()), int(Cs[M[p]][:, M[q]].sum()), float(frac)

    def effective(p, q):  # signed input fraction after 1..MAX_HOPS synapses
        v = M[p].astype(float)
        res = []
        for _ in range(MAX_HOPS):
            v = Wt @ v
            res.append(float(v[M[q]].mean()))
        return res

    def mirror(p, q):
        return MIRROR.get(p, p), MIRROR.get(q, q)

    groups = {k: dict(label=g['label'], neurons=int(M[k].sum()),
                      types=sorted(set(a['cell_type'].values[M[k]]) - {''})[:12]) for k, g in GROUPS.items()}
    for k in groups:
        print(f'{k:5} {groups[k]["neurons"]:6d} neurons  {GROUPS[k]["label"]}', file=sys.stderr)

    # 1) all direct feed-forward connections between brain nodes
    stats = {}
    for p in GROUPS:
        for q in GROUPS:
            if p == q or LAYER[q] == 0 or LAYER[q] < LAYER[p]:
                continue
            stats[(p, q)] = direct(p, q)

    edges = {}
    for (p, q), (n, ns, f) in stats.items():
        mp = mirror(p, q)
        f_sym = (f + stats[mp][2]) / 2 if mp in stats else f   # left/right average
        if n >= MIN_SYN and abs(f_sym) >= MIN_FRAC:
            edges[(p, q)] = dict(kind='direct', hops=1, syn=n, signed=ns, frac=f_sym)

    # 2) model edges without a direct connection: look for a 2-3 synapse pathway
    for p, q in MODEL_EDGES:
        if (p, q) in edges:
            continue
        eff = effective(p, q)
        eff_m = effective(*mirror(p, q)) if mirror(p, q) != (p, q) else eff
        eff = [(x + y) / 2 for x, y in zip(eff, eff_m)]
        hop = next((h for h, x in enumerate(eff, 1) if h > 1 and abs(x) >= MIN_EFF), None)
        n, ns, _ = stats.get((p, q), direct(p, q))
        if hop is None:
            print(f'  drop {p}->{q}: no pathway within {MAX_HOPS} synapses {eff}', file=sys.stderr)
            continue
        edges[(p, q)] = dict(kind='indirect', hops=hop, syn=n, signed=ns, frac=eff[hop - 1])

    for (p, q), d in edges.items():
        d['w'] = round(float(np.sign(d['frac']) * GAIN * abs(d['frac']) ** EXPONENT), 2)

    # 3) olfactory steering. ORN -> DN runs through 3 synapses. The data give the sign and the
    #    relative strength of each pathway (ipsilateral DN, contralateral DN, forward DNs);
    #    the ipsilateral gain is a model parameter and the others are scaled by the data ratio.
    def both_sides(p, q):
        return float(np.mean([effective(p, q)[2], effective(*mirror(p, q))[2]]))
    ipsi, contra, fwd = both_sides('antL', 'dnL'), both_sides('antL', 'dnR'), both_sides('antL', 'dnF')
    for s, o in (('L', 'R'), ('R', 'L')):
        edges[(f'ant{s}', f'dn{s}')] = dict(kind='steer', hops=3, frac=ipsi, w=ODOR_STEER_GAIN)
        edges[(f'ant{s}', 'dnF')] = dict(kind='steer', hops=3, frac=fwd,
                                         w=round(float(ODOR_STEER_GAIN * fwd / ipsi), 2))
        # FlyWire: the contralateral pathway is net excitatory, just weaker than the ipsilateral
        # one. A 16-node rate model saturates if both sides are excited, so the model turns that
        # left/right difference into push-pull inhibition (the old hand-set value).
        edges[(f'ant{s}', f'dn{o}')] = dict(kind='model', hops=3, frac=contra, w=ODOR_CONTRA_W,
                                           note=f'data: net excitatory, {ipsi / contra:.1f}x weaker than ipsilateral')
    print(f'  odor pathways (3 synapses): ipsi {ipsi:.2e}, contra {contra:.2e}, forward {fwd:.2e}', file=sys.stderr)

    # 4) hand-set parts
    for p, q, w, why in HAND_SET:
        edges[(p, q)] = dict(kind='model', w=w, note=why)

    order = list(GROUPS) + ['wL', 'leg', 'wR']
    rows = sorted(edges.items(), key=lambda kv: (order.index(kv[0][0]), order.index(kv[0][1])))
    for (p, q), d in rows:
        extra = f"syn={d['syn']:>7} " if 'syn' in d else ' ' * 12
        print(f"  {p:>5} -> {q:<5} {d['kind']:8} w={d['w']:+5.2f} {extra}{d.get('frac', '')}", file=sys.stderr)

    out = dict(
        source='FlyWire v783 (Dorkenwald et al. 2024, Schlegel et al. 2024); sign from Eckstein et al. 2024 via Shiu et al. 2024',
        rule=dict(gain=GAIN, exponent=EXPONENT, min_frac=MIN_FRAC, min_syn=MIN_SYN,
                  odor_steer_gain=ODOR_STEER_GAIN, odor_contra_w=ODOR_CONTRA_W),
        groups=groups,
        edges=[dict(pre=p, post=q, **d) for (p, q), d in rows],
    )
    (HERE / 'flywire_brain.json').write_text(json.dumps(out, indent=1, ensure_ascii=False))

    # 5) patch index.html
    kind_code = dict(direct='d', indirect='i', steer='s', model='m')
    lines = []
    for (p, q), d in rows:
        lines.append(f"  ['{p}', '{q}', {d['w']}, '{kind_code[d['kind']]}', {d.get('syn', 0)}, {d.get('hops', 0)}],")
    info = {k: dict(n=v['neurons'], label=v['label']) for k, v in groups.items()}
    for k in ('wL', 'leg', 'wR'):
        info[k] = dict(n=0, label='נוירונים מוטוריים בחוט העצבי הגחוני (VNC), שאינו חלק מ-FlyWire')
    block = ('/* FLYWIRE:BEGIN — generated by data/build_brain.py, do not edit by hand */\n'
             '// [pre, post, weight, kind, synapses, hops]\n'
             "// kind: d = direct FlyWire synapses, i = 2-3 synapse FlyWire pathway, s = olfactory steering (FlyWire sign + ratio), m = model assumption\n"
             'const SYN = [\n' + '\n'.join(lines) + '\n];\n'
             f'const FLYWIRE = {json.dumps(info, ensure_ascii=False)};\n'
             '/* FLYWIRE:END */')
    html = SITE.read_text()
    html, n = re.subn(r'/\* FLYWIRE:BEGIN.*?/\* FLYWIRE:END \*/', lambda _: block, html, flags=re.S)
    if n != 1:
        sys.exit('FLYWIRE markers not found in index.html')
    SITE.write_text(html)
    print(f'wrote {len(rows)} connections to index.html and data/flywire_brain.json', file=sys.stderr)


if __name__ == '__main__':
    main()
