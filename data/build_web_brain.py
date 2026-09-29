#!/usr/bin/env python3
"""Pack the whole FlyWire v783 brain into one compact file for the in-browser simulator.

Output: brain/flywire783.bin.gz, read by brain/fullbrain.js. Same neurons (138,639),
connections (15,091,983), signs and neuron groups as brain_server/server.py (Shiu et al. 2024).

How it gets small (~22 MB instead of the 100 MB parquet):
  * neurons are renumbered so that connected neurons sit close together (sorted by
    super class, side, class, hemilineage, cell type, position). Each neuron's targets are
    stored sorted, as differences from the previous target, which are then mostly small;
  * the sign (excitatory / inhibitory) is a property of the presynaptic neuron, so it is one
    byte per neuron instead of one per connection; connections keep only the synapse count;
  * numbers are split into byte planes (all low bytes, then all middle bytes, ...), which
    gzip compresses much better than interleaved integers.

Layout inside the gzip (little endian):
  "ZVUVBRN1" | u32 header length | header JSON (utf-8) | zero padding to a multiple of 4 |
  outdeg u32[n] | type u16[n] | sign i8[n] | side u8[n] | group u8[n] | drive u8[n] |
  target delta planes u8[m] x3 | (synapse count - 1) planes u8[m] x2
Run:  python data/build_web_brain.py
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_brain as bb  # noqa: E402

OUT = HERE.parent / 'brain' / 'flywire783.bin.gz'
PARAMS = dict(v_0=-52.0, v_rst=-52.0, v_th=-45.0, t_mbr=20.0, tau=5.0, t_rfc=2.2, t_dly=1.8,
              w_syn=0.275, f_poi=250, dt=0.1)  # mV and ms, as in Shiu et al. 2024 model.py
SIDES = {'': 0, 'left': 1, 'right': 2, 'center': 3}


def main():
    bb.fetch()
    comp = pd.read_csv(bb.RAW / 'Completeness_783.csv', index_col=0)
    ann = pd.read_csv(bb.RAW / 'Supplemental_file1_neuron_annotations.tsv', sep='\t', dtype=str)
    ann = ann.drop_duplicates('root_id').set_index('root_id').reindex(comp.index.astype(str)).fillna('')
    con = pd.read_parquet(bb.RAW / 'Connectivity_783.parquet',
                          columns=['Presynaptic_Index', 'Postsynaptic_Index', 'Connectivity', 'Excitatory'])
    n, m = len(comp), len(con)

    # --- renumber: connected neurons close together -> small target deltas
    pos_x = pd.to_numeric(ann['pos_x'], errors='coerce').fillna(0).values
    perm = np.lexsort((pos_x, ann['cell_type'].values, ann['ito_lee_hemilineage'].values,
                       ann['cell_class'].values, ann['side'].values, ann['super_class'].values))
    new = np.empty(n, np.int64)
    new[perm] = np.arange(n)          # new[old index] = new index

    pre, post = new[con['Presynaptic_Index'].values], new[con['Postsynaptic_Index'].values]
    cnt = con['Connectivity'].values.astype(np.int64)
    exc = con['Excitatory'].values
    order = np.lexsort((post, pre))
    pre, post, cnt, exc = pre[order], post[order], cnt[order], exc[order]

    sign = np.zeros(n, np.int8)
    sign[pre] = exc
    assert (sign[pre] == exc).all(), 'sign must be a property of the presynaptic neuron'
    outdeg = np.bincount(pre, minlength=n).astype(np.uint32)
    first = np.concatenate([[0], np.cumsum(outdeg, dtype=np.int64)[:-1]])
    delta = post.copy()
    delta[1:] -= post[:-1]
    has = outdeg > 0
    delta[first[has]] = post[first[has]]           # first target of each row is absolute
    assert delta.min() >= 0 and delta.max() < 2 ** 24 and cnt.min() >= 1 and cnt.max() <= 2 ** 16

    # --- neuron info in the new order
    types_old = ann['cell_type'].values
    uniq = sorted(set(types_old) - {''})
    tid = {t: i + 1 for i, t in enumerate(uniq)}   # 0 = no type
    type_idx = np.array([tid.get(t, 0) for t in types_old], np.uint16)[perm]
    side = np.array([SIDES.get(s, 0) for s in ann['side'].values], np.uint8)[perm]

    masks = bb.masks(ann)
    group = np.zeros(n, np.uint8)
    for gi, g in enumerate(bb.REPORT, 1):
        assert not group[new[np.flatnonzero(masks[g])]].any(), f'group {g} overlaps another group'
        group[new[np.flatnonzero(masks[g])]] = gi
    drive = np.zeros(n, np.uint8)
    for si, s in enumerate(bb.SENSORS, 1):
        sel = masks[s] & np.isin(types_old, bb.INPUT_TYPES[s])
        drive[new[np.flatnonzero(sel)]] = si

    header = dict(n=n, m=m, params=PARAMS, types=[''] + uniq, sides=list(SIDES),
                  groups=list(bb.REPORT), sensors=list(bb.SENSORS),
                  source='FlyWire v783 (Dorkenwald et al. 2024, Schlegel et al. 2024); '
                         'connectivity and signs as packaged by Shiu et al. 2024')
    hj = json.dumps(header, ensure_ascii=False, separators=(',', ':')).encode()
    head = b'ZVUVBRN1' + np.uint32(len(hj)).tobytes() + hj
    head += b'\0' * (-len(head) % 4)
    planes = lambda a, k: b''.join(((a >> (8 * i)) & 0xff).astype(np.uint8).tobytes() for i in range(k))
    body = (outdeg.tobytes() + type_idx.tobytes() + sign.tobytes() + side.tobytes() + group.tobytes()
            + drive.tobytes() + planes(delta, 3) + planes(cnt - 1, 2))
    OUT.parent.mkdir(exist_ok=True)
    raw = head + body
    OUT.write_bytes(gzip.compress(raw, 9, mtime=0))
    print(f'{n} neurons, {m} connections -> {OUT.relative_to(HERE.parent)}: '
          f'{len(raw) / 1e6:.1f} MB raw, {OUT.stat().st_size / 1e6:.1f} MB gzip', file=sys.stderr)


if __name__ == '__main__':
    main()
