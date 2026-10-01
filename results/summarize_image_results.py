"""Mean, standard deviation and range over seeds of the image results."""
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def last(d):
    if not d:
        return None
    key = max(d, key=lambda k: int(re.findall(r'\d+', k)[0]))
    return d[key]


def load(setting):
    rows = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(HERE, setting, '*_metrics.json'))):
        arm = re.sub(r'_seed\d+_metrics\.json$', '', os.path.basename(f))
        m = json.load(open(f))
        probe = last(m.get('probe_acc', {}))
        rank = last(m.get('erank', {}))
        row = {'linear': probe['test_acc'] if probe else float('nan'),
               'erank': rank if rank is not None else float('nan')}
        pf = f.replace('_metrics.json', '_probes.json')
        if os.path.exists(pf):
            p = json.load(open(pf))
            row['knn'] = p.get('knn20_test_acc', float('nan'))
            row['attentive'] = p.get('attentive_test_acc', float('nan'))
        rows[arm].append(row)
    return rows


def fmt(xs):
    xs = np.asarray([x for x in xs if x == x], dtype=float)
    if len(xs) == 0:
        return '-'
    sd = xs.std(ddof=1) if len(xs) > 1 else 0.0
    return f'{xs.mean():.4f} ± {sd:.4f} [{xs.min():.4f}, {xs.max():.4f}]'


def main():
    settings = sorted({os.path.relpath(os.path.dirname(f), HERE)
                       for f in glob.glob(os.path.join(HERE, '**', '*_metrics.json'), recursive=True)})
    for setting in settings:
        print(f'\n== {setting}')
        for arm, rows in sorted(load(setting).items()):
            line = f'  {arm:<28} n={len(rows)}  linear {fmt([r["linear"] for r in rows])}'
            line += f'  erank {fmt([r["erank"] for r in rows])}'
            if any('knn' in r for r in rows):
                line += f'  knn {fmt([r.get("knn", float("nan")) for r in rows])}'
                line += f'  attentive {fmt([r.get("attentive", float("nan")) for r in rows])}'
            print(line)


if __name__ == '__main__':
    main()
