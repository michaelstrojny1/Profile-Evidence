#!/usr/bin/env python3
"""
Cleanup outputs directory: keep the latest overnight_benchmark_optimized_* and
the latest ten_configs_* directories; delete everything else under outputs/.

Usage:
  python -u scripts/cleanup_outputs.py
"""

import os
import glob
import shutil
from datetime import datetime


def latest_by_glob(pattern: str) -> str | None:
    paths = glob.glob(pattern)
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p))
    return paths[-1]


def main() -> None:
    root = 'outputs'
    if not os.path.isdir(root):
        print('No outputs/ directory present; nothing to clean.')
        return

    keep_dirs = set()
    latest_overnight = latest_by_glob(os.path.join(root, 'overnight_benchmark_optimized_*'))
    latest_ten = latest_by_glob(os.path.join(root, 'ten_configs_*'))
    if latest_overnight:
        keep_dirs.add(os.path.basename(latest_overnight))
    if latest_ten:
        keep_dirs.add(os.path.basename(latest_ten))

    print('Keeping directories:')
    for k in sorted(keep_dirs):
        print(f'  - {k}')

    removed = []
    for name in os.listdir(root):
        path = os.path.join(root, name)
        if name in keep_dirs:
            continue
        try:
            if os.path.isdir(path):
                shutil.rmtree(path, ignore_errors=True)
                removed.append(name + os.sep)
            else:
                os.remove(path)
                removed.append(name)
        except Exception as e:
            print(f'WARN: failed to remove {path}: {e}')

    print('\nRemoved:')
    for r in removed:
        print(f'  - {r}')
    print('\nCleanup complete.')


if __name__ == '__main__':
    main()


