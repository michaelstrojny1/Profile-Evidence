#!/usr/bin/env python3
"""
Check PL variability in saved *_pe_results.pkl files.

Usage:
  python -u scripts/check_pl_variation.py [PKL ...]
If no PKL args are passed, scans under 'outputs' for '*_pe_results.pkl'.
Prints min/max/range with high precision.
"""

import os
import sys
import glob
import pickle
import numpy as np


def iter_pkls(paths):
    if paths:
        for p in paths:
            if os.path.isdir(p):
                for q in glob.glob(os.path.join(p, '**', '*_pe_results.pkl'), recursive=True):
                    yield q
            elif os.path.isfile(p):
                yield p
            else:
                for q in glob.glob(p):
                    if os.path.isfile(q):
                        yield q
    else:
        for q in glob.glob(os.path.join('outputs', '**', '*_pe_results.pkl'), recursive=True):
            yield q


def main():
    pkls = list(iter_pkls(sys.argv[1:]))
    if not pkls:
        print('No *_pe_results.pkl files found.')
        return

    print(f'Checking {len(pkls)} files...')
    for p in pkls:
        try:
            with open(p, 'rb') as f:
                res = pickle.load(f)
            pl = np.asarray(res.get('pl_vals'), dtype=float)
            pe = np.asarray(res.get('pe_vals'), dtype=float)
            w = np.asarray(res.get('omegas'), dtype=float)
            if pl.size == 0:
                print(f'- {p}: pl_vals missing/empty')
                continue
            pl_min = float(np.min(pl))
            pl_max = float(np.max(pl))
            pl_rng = float(pl_max - pl_min)
            pe_min = float(np.min(pe)) if pe.size else float('nan')
            pe_max = float(np.max(pe)) if pe.size else float('nan')
            pe_rng = float(pe_max - pe_min) if pe.size else float('nan')
            print(f'- {p}')
            print(f'    PL: min={pl_min:.8f} max={pl_max:.8f} range={pl_rng:.8e} (n={pl.size})')
            print(f'    PE: min={pe_min:.8f} max={pe_max:.8f} range={pe_rng:.8e} (n={pe.size})')
            if pl_rng == 0.0:
                print('    NOTE: PL identical to machine precision across grid.')
        except Exception as e:
            print(f'- {p}: FAILED to read ({e})')


if __name__ == '__main__':
    main()


