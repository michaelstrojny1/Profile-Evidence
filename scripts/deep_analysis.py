#!/usr/bin/env python3
"""
Deep analysis over the latest outputs/overnight_benchmark_optimized_* directory.
Computes compensation metrics, ESS, PL/PE variability, CI summaries, and
ridge-volume correlations. Writes deep_analysis.txt in that directory and
prints a short recap.
"""

import os
import glob
import json
import pickle
import numpy as np
from datetime import datetime


def latest_overnight_dir(root: str = 'outputs') -> str | None:
    paths = glob.glob(os.path.join(root, 'overnight_benchmark_optimized_*'))
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p))
    return paths[-1]


def ci_mid(ci: tuple[float, float]) -> float:
    lo, hi = ci
    if np.isnan(lo) or np.isnan(hi):
        return np.nan
    return 0.5 * (lo + hi)


def safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    try:
        if x.size < 2 or y.size < 2:
            return np.nan
        return float(np.corrcoef(x, y)[0, 1])
    except Exception:
        return np.nan


def main() -> None:
    out_dir = latest_overnight_dir()
    if not out_dir:
        print('No overnight_benchmark_optimized_* directory found under outputs/.')
        return

    rows = []
    for fname in sorted(os.listdir(out_dir)):
        if not fname.endswith('_pe_results.pkl'):
            continue
        with open(os.path.join(out_dir, fname), 'rb') as f:
            res = pickle.load(f)

        cfg = res.get('config', {})
        model = res.get('model', 'smib')
        seed = res.get('seed', 0)
        comp = bool(cfg.get('enable_comp_demo', False))
        noise = cfg.get('zeta_noise_dist', 'na')
        delta = np.array(res.get('delta_norm', []), float)
        ess = np.array(res.get('ess_rels', []), float)
        pl = np.array(res.get('pl_vals', []), float)
        pe = np.array(res.get('pe_vals', []), float)
        pl_ci = res.get('pl_ci', (np.nan, np.nan))
        pe_ci = res.get('pe_ci', (np.nan, np.nan))
        comp_gains = np.array(res.get('comp_gains', []), float)
        ridge = np.array(res.get('ridge_logvols', []), float)

        rows.append(dict(
            file=fname,
            model=model,
            seed=int(seed),
            comp=comp,
            noise=noise,
            pos_frac=float(np.mean(delta > 0.05)) if delta.size else np.nan,
            neg_frac=float(np.mean(delta < -0.05)) if delta.size else np.nan,
            pos_count=int(np.sum(delta > 0.05)) if delta.size else 0,
            neg_count=int(np.sum(delta < -0.05)) if delta.size else 0,
            ess_mean=float(np.mean(ess)) if ess.size else np.nan,
            ess_min=float(np.min(ess)) if ess.size else np.nan,
            pl_range=float(np.max(pl) - np.min(pl)) if pl.size else np.nan,
            pe_range=float(np.max(pe) - np.min(pe)) if pe.size else np.nan,
            pl_ci=str(pl_ci), pe_ci=str(pe_ci),
            pl_mid=float(ci_mid(pl_ci)), pe_mid=float(ci_mid(pe_ci)),
            gain_vs_ridge=float(safe_corr(comp_gains, ridge)),
        ))

    # Aggregate by regime
    comp_on = [r for r in rows if r['comp']]
    comp_off = [r for r in rows if not r['comp']]

    def agg(stats):
        if not stats:
            return {}
        def m(name):
            vals = [r[name] for r in stats if isinstance(r[name], (int, float)) and not np.isnan(r[name])]
            return float(np.mean(vals)) if vals else np.nan
        return {
            'pos_frac_mean': m('pos_frac'),
            'neg_frac_mean': m('neg_frac'),
            'ess_mean': m('ess_mean'),
            'ess_min_mean': m('ess_min'),
            'pl_range_mean': m('pl_range'),
            'pe_range_mean': m('pe_range'),
            'gain_vs_ridge_mean': m('gain_vs_ridge'),
        }

    summary = {
        'generated': datetime.now().isoformat(timespec='seconds'),
        'out_dir': out_dir,
        'n_runs': len(rows),
        'by_run': rows,
        'comp_on': agg(comp_on),
        'comp_off': agg(comp_off),
        'pos_frac_effect': (agg(comp_on).get('pos_frac_mean', np.nan) - agg(comp_off).get('pos_frac_mean', np.nan)) if rows else np.nan,
    }

    # Write
    out_txt = os.path.join(out_dir, 'deep_analysis.txt')
    out_json = os.path.join(out_dir, 'deep_analysis.json')
    lines = []
    lines.append('DEEP ANALYSIS SUMMARY')
    lines.append('=' * 80)
    lines.append(f"Directory: {out_dir}")
    lines.append(f"Generated: {summary['generated']}")
    lines.append('')
    for r in rows:
        lines.append(f"- {r['file']}: comp={r['comp']} noise={r['noise']} | posFrac={r['pos_frac']:.3f} negFrac={r['neg_frac']:.3f} | ESS(mean/min)={r['ess_mean']:.3f}/{r['ess_min']:.3f} | PLrng={r['pl_range']:.3e} PErng={r['pe_range']:.3f} | gain~ridge={r['gain_vs_ridge']:.3f}")
    lines.append('')
    lines.append(f"Comp ON mean posFrac: {summary['comp_on'].get('pos_frac_mean', np.nan):.3f}")
    lines.append(f"Comp OFF mean posFrac: {summary['comp_off'].get('pos_frac_mean', np.nan):.3f}")
    if not np.isnan(summary['pos_frac_effect']):
        lines.append(f"Compensation effect (posFrac_ON - posFrac_OFF): {summary['pos_frac_effect']:+.3f}")

    with open(out_txt, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))
    with open(out_json, 'w', encoding='utf-8') as fh:
        json.dump(summary, fh, indent=2)
    print('\n'.join(lines))
    print(f"\nSaved: {out_txt}\nSaved: {out_json}")


if __name__ == '__main__':
    main()


