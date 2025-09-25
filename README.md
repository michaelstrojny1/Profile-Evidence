# Profile Evidence

## Overview

Comparison of **Profile Likelihood (PL)** and **Profile Evidence (PE)** for parameter inference across multiple physical systems. Companion to this paper:

## Summary

PE corrects PL's underestimated likelihood when nuisance parameters can compensate for the profiled parameter. This occurs through:

1. **Time-varying nuisance amplitude**: ζ(t) follows AR(1) dynamics + jitter
2. **Correlated process noise**: Non-diagonal Q in simulation (diagonal in estimator)
3. **Anisotropic measurements**: Higher noise on second observation channel
4. **Model mismatch**: Creates compensation opportunities where PE > PL

### Diagnostics
- **Compensation gain**: Laplace evidence - PL
- **Ridge log-volume**: Measures nuisance parameter ridge width
- **Delta curve**: PE_norm - PL_norm
- **ESS monitoring**: Ensures importance sampling reliability

## Test Data
```
2D Damped Oscillator: ẍ + γẋ + ω²x = ζω sin(ωt) + noise
- Parameter of interest: ω (angular frequency)
- Nuisance parameters: γ (damping), ζ (forcing amplitude)
- State estimation: Kalman filter for linear observations
```

SMIB Small-Signal (linearized about operating point):
```
x = [Δδ, Δω]ᵀ
Δδ̇ = Δω
Δω̇ = ( -K cos(δ₀) Δδ - D Δω ) / M + w_t
y = x + v_t
```
- Parameter of interest: K (effective electrical stiffness)
- Nuisance: D (damping), δ₀ (operating angle); inertia M known
- Compensation: simulation uses time-varying δ₀(t) (AR(1)+jitter), estimator assumes constant δ₀

### Algorithm Flow
1. **Simulate** compensation scenario with model mismatches
2. **Coarse grid** (33 points) to find PL peak
3. **Refined grid** (21 points) around peak with guaranteed true ω coverage
4. **For each ω**:
   - Compute PL via MLE optimization
   - Find MAP and stabilized Hessian for PE
   - Grid search proposal scales, select highest ESS
   - Estimate PE via raw importance sampling
   - Record compensation diagnostics
5. **Normalize** curves and compute confidence intervals
6. **Generate** diagnostic plots and summaries

### Configuring Tests
- Toggle compensation regime: `CFG.enable_comp_demo ∈ {True, False}`
- Nuisance amplitude noise distributions for ζ dynamics: `CFG.zeta_noise_dist ∈ {'gaussian','laplace','student_t'}` with `CFG.zeta_t_nu` (default 4.0)
- Multi-seed evaluation: seeds × regimes produce aggregated coverage/width/correlation metrics

For SMIB (`pe_SMIB.py`), the same sweeps apply with ζ renamed to δ₀.

## Results

Please see **results directory** and **REPRODUCABILITY.md**. There are too many to include here.

#### Some Quick Visuals (Oscillator vs SMIB)

Oscillator (compensation ON):

![Oscillator PL vs PE](readme_assets/oscillator_s0_True_gaussian_bulletproof_pe_results.png)

![Oscillator Delta](readme_assets/oscillator_s0_True_gaussian_pl_pe_delta.png)

Compensation vs Overestimation clouds:

![Compensation cloud](readme_assets/oscillator_s0_True_gaussian_compensation_cloud_omega_1.845.png)
![Overestimation cloud](readme_assets/oscillator_s0_True_gaussian_overestimation_cloud_omega_1.965.png)

SMIB (control, compensation OFF):

![SMIB PL vs PE](readme_assets/smib_s3_False_gaussian_bulletproof_pe_results.png)

![SMIB Delta](readme_assets/smib_s3_False_gaussian_pl_pe_delta.png)

### What the Figures Show (and how to read them)

- PL vs PE curves: Normalized log-scales for fair comparison using a shared χ²(1) 95% threshold. Bands where PE lies above PL indicate compensation regions where nuisance parameter volume lifts total evidence.
- Delta curve (PE_norm − PL_norm): Positive segments = compensation (PE corrects PL underestimation); negative segments = PL overestimation penalized by evidence.
- Compensation/Overestimation clouds: Weighted importance-sampling clouds at selected ω points, showing exactly which nuisance values carry weight. Ellipse overlays depict 2σ of the local covariance around MAP.
- Predictive checks (not shown here): RMSE vs ω and residual diagnostics at the PL peak (QQ, ACF).

### Plots
- `compensation_path.png`: MAP/MLE nuisance trajectories vs ω showing parameter coupling
- `compensation_cloud_omega_*.png`: Weighted IS sample clouds at high-δ ω points (PE > PL)
- `overestimation_cloud_omega_*.png`: Weighted IS sample clouds at negative-δ ω points (PE < PL)
- Shows exactly which nuisance values (within uncertainty) compensate or over-compensate the profiled parameter

### Summary Files
- `pe_vs_pl_summary.txt`: Performance at true ω, fraction with PE>PL, ESS statistics
- `compensation_summary.txt`: Peak compensation location and statistics
- `correction_flags.txt`: Specific ω points where PE corrects PL (Δ>0.05)
- `overestimation_flags.txt`: Specific ω points where PE indicates PL overestimation (Δ<-0.05)
- `benchmark_summary.txt`: Multi-seed validation metrics

