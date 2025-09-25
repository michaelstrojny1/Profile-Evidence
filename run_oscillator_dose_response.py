#!/usr/bin/env python3
import os
import time
import pickle
from datetime import datetime
from typing import List, Dict, Any

import pe as osc

STRENGTHS = [0.0, 1.0, 1.5, 2.0]  # 0x control, baseline, enhanced, strong
NOISES = ['gaussian', 'laplace', 'student_t']
SEEDS = [0,1,2,3,4,5]

# Mapping of strength -> multipliers for jitter/walk
def apply_strength(strength: float):
    if strength <= 0.0:
        osc.CFG.enable_comp_demo = False
        osc.CFG.enable_q_corr = False
        osc.CFG.zeta_noise_dist = 'gaussian'
        osc.CFG.zeta_jitter_std = 0.0
        osc.CFG.zeta_walk_std = 0.0
        return
    osc.CFG.enable_comp_demo = True
    osc.CFG.enable_q_corr = True
    factor = strength
    osc.CFG.zeta_jitter_std = 0.30 * factor
    osc.CFG.zeta_walk_std = 0.35 * factor


def run_one(seed: int, strength: float, noise: str, out_dir: str) -> Dict[str, Any]:
    osc.CFG.zeta_noise_dist = noise
    apply_strength(strength)
    prefix = os.path.join(out_dir, f"osc_s{seed}_{strength:g}_{noise}_")
    os.environ['RUN_PREFIX'] = prefix
    res = osc.run_profile_evidence(seed=seed, make_plots=True, write_files=True)
    res['model'] = 'oscillator'
    res['seed'] = seed
    res['config'] = {'strength': strength, 'zeta_noise_dist': noise, 'enable_comp_demo': strength>0}
    return res


def main():
    base = os.environ.get('RUN_DIR_BASE', 'outputs')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(base, f'oscillator_dose_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output: {out_dir}")

    strengths_env = os.environ.get('STRENGTHS')
    if strengths_env:
        strengths = [float(s) for s in strengths_env.split(',')]
    else:
        strengths = STRENGTHS

    seeds_env = os.environ.get('SEEDS')
    seeds = [int(s) for s in seeds_env.split(',')] if seeds_env else SEEDS

    noises_env = os.environ.get('NOISE_LIST')
    noises = noises_env.split(',') if noises_env else NOISES

    total = len(seeds)*len(strengths)*len(noises)
    done = 0
    ok = 0
    start = time.time()
    all_results: List[Dict[str, Any]] = []

    for seed in seeds:
        for strength in strengths:
            for noise in noises:
                try:
                    print(f"[{done+1}/{total}] seed={seed} strength={strength} noise={noise}")
                    res = run_one(seed, strength, noise, out_dir)
                    ok += 1
                    all_results.append(res)
                    with open(os.path.join(out_dir, f"osc_s{seed}_{strength:g}_{noise}.pkl"), 'wb') as f:
                        pickle.dump(res, f)
                except Exception as e:
                    import traceback
                    print(f"  ERROR: {e}")
                    traceback.print_exc()
                finally:
                    done += 1
                    elapsed = time.time() - start
                    eta = elapsed * (total - done) / max(1, done)
                    print(f"Progress: {done}/{total} ok={ok} elapsed={elapsed/3600:.1f}h ETA={eta/3600:.1f}h")
                    print('-'*60)

    with open(os.path.join(out_dir, 'oscillator_dose_results.pkl'), 'wb') as f:
        pickle.dump(all_results, f)
    print("DONE")

if __name__ == '__main__':
    main()
