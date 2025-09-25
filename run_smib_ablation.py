#!/usr/bin/env python3
"""
SMIB-only ablation runner with env-driven configuration.

Environment variables:
- RUN_DIR_BASE: base output directory (default: 'outputs')
- SEEDS: comma-separated seeds (default: '0,1,2,3,4,5')
- COMP_LIST: comma-separated comp flags in {0,1,True,False} (default: '1,0')
- NOISE_LIST: comma-separated noise names (default: 'gaussian,laplace,student_t')
- REFINE_POINTS: refined grid size for PE/PL (default: '25')
- CLOUD_VMAX: optional fixed vmax for clouds ('1'|'true' to fix at 1.0)

This script sets RUN_PREFIX per configuration and calls pe_smib.run_bulletproof.
It also logs CI midpoints and midpoint errors into a CSV in the run directory.
"""

import os
import csv
from datetime import datetime
from typing import List

import numpy as np

import pe_smib as smib


def _parse_list(env_val: str, coercer) -> List:
    return [coercer(x.strip()) for x in env_val.split(',') if x.strip()]


def _to_bool(x: str) -> bool:
    return x in ('1', 'true', 'True')


def main() -> None:
    # Output directory
    run_dir_base = os.environ.get('RUN_DIR_BASE', 'outputs')
    os.makedirs(run_dir_base, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(run_dir_base, f"smib_ablation_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)
    os.environ['RUN_DIR'] = run_dir

    # Config matrix
    seeds = _parse_list(os.environ.get('SEEDS', '0,1,2,3,4,5'), int)
    comp_list = _parse_list(os.environ.get('COMP_LIST', '1,0'), _to_bool)
    noise_list = _parse_list(os.environ.get('NOISE_LIST', 'gaussian,laplace,student_t'), str)

    # Analysis controls
    os.environ.setdefault('REFINE_POINTS', os.environ.get('REFINE_POINTS', '25'))
    if 'CLOUD_VMAX' in os.environ:
        os.environ['CLOUD_VMAX'] = os.environ['CLOUD_VMAX']

    # CSV for midpoint errors
    csv_path = os.path.join(run_dir, 'midpoint_errors.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as fh:
        writer = csv.writer(fh)
        writer.writerow([
            'model', 'seed', 'comp', 'noise',
            'omega_true',
            'pl_ci_lo', 'pl_ci_hi', 'pl_mid', 'pl_mid_err', 'pl_width',
            'pe_ci_lo', 'pe_ci_hi', 'pe_mid', 'pe_mid_err', 'pe_width'
        ])

    total = len(seeds) * len(comp_list) * len(noise_list)
    done = 0
    print(f"SMIB ablation total configurations: {total}")
    print(f"Run directory: {run_dir}")

    for seed in seeds:
        for comp in comp_list:
            for noise in noise_list:
                done += 1
                print(f"\n[{done:02d}/{total}] SMIB seed={seed}, comp={comp}, noise={noise}")

                # Configure model
                smib.CFG.zeta_noise_dist = noise
                if comp:
                    smib.CFG.enable_comp_demo = True
                    # Use defaults for correlation/anisotropy as defined in pe_smib.py
                else:
                    smib.CFG.enable_comp_demo = False
                    smib.CFG.enable_q_corr = False
                    smib.CFG.r_std2 = smib.CFG.r_std

                # RUN_PREFIX per configuration
                os.environ['RUN_PREFIX'] = os.path.join(
                    run_dir,
                    f"smib_s{seed}_{comp}_{noise}_"
                )

                # Execute
                results = smib.run_bulletproof(seed=int(seed), make_plots=True, write_files=True)

                # Midpoint error logging
                try:
                    omega_true = float(results.get('omega_true', smib.CFG.omega_true))
                    pl_ci = results.get('pl_ci', (np.nan, np.nan))
                    pe_ci = results.get('pe_ci', (np.nan, np.nan))

                    def _mid_err(ci):
                        lo, hi = ci
                        if np.isnan(lo) or np.isnan(hi):
                            return np.nan, np.nan, np.nan
                        mid = 0.5 * (lo + hi)
                        return mid, abs(mid - omega_true), (hi - lo)

                    pl_mid, pl_err, pl_w = _mid_err(pl_ci)
                    pe_mid, pe_err, pe_w = _mid_err(pe_ci)

                    with open(csv_path, 'a', newline='', encoding='utf-8') as fh:
                        writer = csv.writer(fh)
                        writer.writerow([
                            'smib', int(seed), bool(comp), noise,
                            omega_true,
                            pl_ci[0], pl_ci[1], pl_mid, pl_err, pl_w,
                            pe_ci[0], pe_ci[1], pe_mid, pe_err, pe_w,
                        ])

                    print(f"  Midpoint errors | PL: {pl_err if not np.isnan(pl_err) else 'nan'} | PE: {pe_err if not np.isnan(pe_err) else 'nan'}")
                except Exception as e:
                    print(f"  WARNING: midpoint error logging failed: {e}")

    print(f"\nSMIB ablation complete. Midpoint errors written to: {csv_path}")


if __name__ == '__main__':
    main()


