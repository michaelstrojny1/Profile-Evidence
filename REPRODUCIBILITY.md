# Reproducibility Guide

## Environment

- Python 3.11+
- Install dependencies:
```bash
pip install -r requirements.txt
```

## Data & Outputs
- All outputs are written under `outputs/` with timestamped directories.
- Each run prefixes artifacts with `RUN_PREFIX` to prevent overwrites.

## Run Validation
- SMIB ablation (seed=0, comp on/off × 3 noise):
```powershell
$env:RUN_DIR_BASE='outputs'; $env:SEEDS='0'; $env:COMP_LIST='1,0'; $env:NOISE_LIST='gaussian,laplace,student_t'; $env:REFINE_POINTS='25'; python -u run_smib_ablation.py
```

- Cross-model validation (oscillator + SMIB):
```powershell
python -u run_crossmodel_validation.py
```

## Consolidate Results
```powershell
python -u scripts/deep_analysis.py outputs
python -u midpoint_error_analysis.py outputs
python -u scripts/statistical_analysis.py outputs
python -u scripts/generate_paper_figures.py outputs paper_figures
```

## Paper
- Final manuscript: `paper.tex`
- Figures copied to `figures/` (PNG):
  - `figures/figure_1_multi_model_comparison.png`
  - `figures/figure_2_statistical_significance.png`

## Notes
- All scripts avoid hardcoding; configuration via environment variables only.
- Artifacts include manifests, flags, clouds, paths, and summaries for direct evidence.
