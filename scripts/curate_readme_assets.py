#!/usr/bin/env python3
import os
import shutil
from glob import glob

ASSETS_DIR = 'readme_assets'
SOURCES = [
    'paper_release/figures/oscillator_s0_True_gaussian_bulletproof_pe_results.png',
    'paper_release/figures/oscillator_s0_True_gaussian_pl_pe_delta.png',
    'paper_release/figures/oscillator_s0_True_gaussian_compensation_cloud_omega_1.845.png',
    'paper_release/figures/oscillator_s0_True_gaussian_overestimation_cloud_omega_1.965.png',
    'paper_release/figures/smib_s3_False_gaussian_bulletproof_pe_results.png',
    'paper_release/figures/smib_s3_False_gaussian_pl_pe_delta.png',
]

def main():
    os.makedirs(ASSETS_DIR, exist_ok=True)
    copied = 0
    for src in SOURCES:
        if os.path.exists(src):
            shutil.copy2(src, ASSETS_DIR)
            copied += 1
        else:
            # try to find by basename anywhere under outputs as fallback
            base = os.path.basename(src)
            for root, _, files in os.walk('outputs'):
                if base in files:
                    shutil.copy2(os.path.join(root, base), ASSETS_DIR)
                    copied += 1
                    break
    print(f"Curated {copied} images into {ASSETS_DIR}")

if __name__ == '__main__':
    main()
