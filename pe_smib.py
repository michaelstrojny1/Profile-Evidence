#!/usr/bin/env python3
"""
Profile Evidence vs Profile Likelihood (SMIB)
Neutral, robust raw-IS implementation with adaptive scaling and diagnostics.
"""

import numpy as np
import os
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import stats
from scipy.optimize import minimize
import pickle

DTYPE = np.float64

def get_run_prefix() -> str:
    return os.environ.get('RUN_PREFIX', '')

class CFG:
    # SMIB small-signal parameters
    # Parameter of interest (effective electrical stiffness K = E'V/X)
    omega_true: float = 2.0  # interpreted as K_true in SMIB
    # Nuisance parameters: damping D and operating angle delta0
    gamma_true: float = 0.5  # D (damping)
    zeta_true: float = 0.7   # delta0 (operating angle, rad), < pi/2
    M_inertia: float = 6.0   # inertia constant M
    
    # Simulation
    dt: float = 0.02
    T_analysis: int = 600
    x0 = np.array([0.05, 0.0], dtype=DTYPE)  # small ring-down to increase observability
    q_std: float = 0.08
    r_std: float = 0.05
    r_std2: float = 0.40  # higher noise on second channel to reduce identifiability
    
    # Priors (weakly informative, physically plausible)
    # [gamma=D, zeta=delta0]
    prior_mean = np.array([0.6, 0.8], dtype=DTYPE)
    prior_cov = np.array([[0.15**2, 0.0], [0.0, 0.25**2]], dtype=DTYPE)
    bounds = [(0.05, 3.0), (0.2, 1.45)]  # D in [0.05,3.0], delta0 in [0.2,1.45]
    
    # Compensation demo (makes operating angle vary in-time to create compensation opportunity)
    enable_comp_demo: bool = True
    # Nuisance amplitude noise distribution for delta0 dynamics ('gaussian' | 'laplace' | 'student_t')
    zeta_noise_dist: str = 'gaussian'
    zeta_t_nu: float = 4.0  # degrees of freedom for Student-t (nu>2)
    zeta_jitter_std: float = 0.20
    zeta_ar1_rho: float = 0.92
    zeta_walk_std: float = 0.20
    enable_q_corr: bool = True
    q_corr: float = 0.5  # correlation between process noise dims in simulation only

    # Known excitation (matched in inference) to improve identifiability
    # input_type: 'none' | 'step' | 'chirp'
    input_type: str = 'step'
    input_amp: float = 0.02
    input_t0: int = 100  # step start index (samples)

def A_matrix(kappa: float, damping: float, dt: float, delta0: float, M: float) -> np.ndarray:
    """State transition for SMIB small-signal linearized around (delta0, omega=0).
    x = [Delta delta, Delta omega]^T;  delta_dot = omega; omega_dot = ( -kappa*cos(delta0)*Delta delta - damping*Delta omega ) / M
    Discretize with forward Euler.
    """
    k_lin = (kappa * np.cos(delta0)) / M
    d_lin = damping / M
    return np.array([
        [1.0, dt],
        [-k_lin * dt, 1.0 - d_lin * dt]
    ], dtype=DTYPE)

def _draw_noise(rng: np.random.Generator, std: float, dist: str, t_nu: float) -> float:
    if std <= 0.0:
        return 0.0
    if dist == 'gaussian':
        return float(rng.normal(0.0, std))
    if dist == 'laplace':
        b = std / np.sqrt(2.0)
        return float(rng.laplace(0.0, b))
    if dist == 'student_t':
        nu = max(2.1, float(t_nu))
        scale = std / np.sqrt(nu / (nu - 2.0))
        return float(rng.standard_t(nu) * scale)
    return float(rng.normal(0.0, std))

def simulate(omega: float, gamma: float, zeta: float, T: int, dt: float,
             x0: np.ndarray, q_std: float, r_std: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Simulate SMIB small-signal linearized model with time-varying operating angle (comp demo)."""
    rng = np.random.default_rng(seed)
    
    xs = np.zeros((T, 2), dtype=DTYPE)
    ys = np.zeros((T, 2), dtype=DTYPE)
    
    x = x0.copy()
    # Correlated process noise (only in data generation; filter still assumes diagonal Q)
    if CFG.enable_q_corr and abs(CFG.q_corr) > 0.0:
        Qchol = np.linalg.cholesky(np.array([[1.0, CFG.q_corr],[CFG.q_corr, 1.0]], dtype=DTYPE))
    else:
        Qchol = np.eye(2, dtype=DTYPE)
    zeta_eff = zeta
    for t in range(T):
        # Time-varying operating angle (delta0) to create compensation via kappa*cos(delta0)
        if CFG.enable_comp_demo:
            zeta_eff = (CFG.zeta_ar1_rho * zeta_eff
                        + (1.0 - CFG.zeta_ar1_rho) * zeta
                        + _draw_noise(rng, CFG.zeta_walk_std, CFG.zeta_noise_dist, CFG.zeta_t_nu))
            zeta_eff = zeta_eff + _draw_noise(rng, CFG.zeta_jitter_std, CFG.zeta_noise_dist, CFG.zeta_t_nu)
        # Known excitation (mechanical torque disturbance) for identifiability
        u_t = 0.0
        if CFG.input_type == 'step' and t >= CFG.input_t0:
            u_t = CFG.input_amp
        elif CFG.input_type == 'chirp':
            # Slow chirp over the window
            f0, f1 = 0.1, 1.0
            tau = t / max(1, T-1)
            freq = f0 * (f1/f0) ** tau
            u_t = CFG.input_amp * np.sin(2*np.pi*freq*(t*dt))

        # Dynamics: x_{t+1} = A x_t + B u_t + w_t, with B=[0; 1/M]
        A_t = A_matrix(omega, gamma, dt, zeta_eff, CFG.M_inertia)
        eps = rng.normal(0, 1.0, 2).astype(DTYPE)
        proc_noise = (Qchol @ eps) * q_std
        B = np.array([0.0, dt/CFG.M_inertia], dtype=DTYPE)
        x = A_t @ x + B * u_t + proc_noise
        # Measurement with anisotropic noise (channel 2 noisier)
        y = x + np.array([rng.normal(0, r_std, 1)[0], rng.normal(0, CFG.r_std2, 1)[0]], dtype=DTYPE)
        
        xs[t] = x
        ys[t] = y
    
    return xs, ys

def kalman_loglik(y: np.ndarray, omega: float, gamma: float, zeta: float, dt: float,
                  q_var: float, r_var: float) -> float:
    """Kalman filter predictive log-likelihood for SMIB small-signal linear model (constant delta0)."""
    T = y.shape[0]
    A = A_matrix(omega, gamma, dt, zeta, CFG.M_inertia)
    Q = q_var * np.eye(2, dtype=DTYPE)
    # Use anisotropic measurement covariance matching simulation (still known to estimator)
    R = np.array([[r_var, 0.0],[0.0, (CFG.r_std2**2)]], dtype=DTYPE)
    
    x = np.zeros(2, dtype=DTYPE)
    P = 0.2 * np.eye(2, dtype=DTYPE)
    ll = 0.0
    
    for t in range(T):
        # Known excitation model (matched to simulation): B u_t with same u_t as in simulate
        u_t = 0.0
        if CFG.input_type == 'step' and t >= CFG.input_t0:
            u_t = CFG.input_amp
        elif CFG.input_type == 'chirp':
            f0, f1 = 0.1, 1.0
            tau = t / max(1, T-1)
            freq = f0 * (f1/f0) ** tau
            u_t = CFG.input_amp * np.sin(2*np.pi*freq*(t*dt))
        B = np.array([0.0, dt/CFG.M_inertia], dtype=DTYPE)

        # Predict
        x_pred = A @ x + B * u_t
        P_pred = A @ P @ A.T + Q
        
        # Update
        S = P_pred + R
        try:
            L = np.linalg.cholesky(S)
            K = P_pred @ np.linalg.solve(S, np.eye(2, dtype=DTYPE))
            innov = y[t] - x_pred
            x = x_pred + K @ innov
            P = (np.eye(2, dtype=DTYPE) - K) @ P_pred
            
            alpha = np.linalg.solve(L, innov)
            ll += -0.5 * (np.sum(alpha**2) + 2.0 * np.sum(np.log(np.diag(L))) + 2.0 * np.log(2.0 * np.pi))
        except np.linalg.LinAlgError:
            return -1e10
    
    return float(ll)

def log_prior_psi(psi: np.ndarray) -> float:
    """Log prior for nuisance parameters."""
    try:
        return float(stats.multivariate_normal.logpdf(psi, CFG.prior_mean, CFG.prior_cov))
    except:
        return -1e10

def log_joint(y: np.ndarray, omega: float, psi: np.ndarray, dt: float,
              q_var: float, r_var: float) -> float:
    """Log joint density."""
    gamma, zeta = float(psi[0]), float(psi[1])
    ll = kalman_loglik(y, omega, gamma, zeta, dt, q_var, r_var)
    lp = log_prior_psi(psi)
    return float(ll + lp)

def find_mle(y: np.ndarray, omega: float, psi0: np.ndarray, dt: float,
             q_var: float, r_var: float) -> tuple[np.ndarray, float]:
    """Find MLE for nuisance parameters."""
    def neg_ll(psi):
        gamma, zeta = float(psi[0]), float(psi[1])
        return -kalman_loglik(y, omega, gamma, zeta, dt, q_var, r_var)
    
    res = minimize(neg_ll, psi0, bounds=CFG.bounds, method='L-BFGS-B')
    return res.x.astype(DTYPE), float(-res.fun)

def find_map(y: np.ndarray, omega: float, psi0: np.ndarray, dt: float,
             q_var: float, r_var: float) -> tuple[np.ndarray, float]:
    """Find MAP for nuisance parameters."""
    def neg_lp(psi):
        return -log_joint(y, omega, psi, dt, q_var, r_var)
    
    res = minimize(neg_lp, psi0, bounds=CFG.bounds, method='L-BFGS-B')
    return res.x.astype(DTYPE), float(-res.fun)

def hessian_fd(f, x, h=1e-5):
    """Finite difference Hessian."""
    n = len(x)
    H = np.zeros((n, n), dtype=DTYPE)
    
    for i in range(n):
        for j in range(n):
            x_pp = x.copy(); x_pp[i] += h; x_pp[j] += h
            x_pm = x.copy(); x_pm[i] += h; x_pm[j] -= h
            x_mp = x.copy(); x_mp[i] -= h; x_mp[j] += h
            x_mm = x.copy(); x_mm[i] -= h; x_mm[j] -= h
            
            H[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4.0 * h * h)
    
    return H

def stabilize_cov(cov: np.ndarray, min_eig: float = 1e-6, max_cond: float = 1e6) -> np.ndarray:
    """Stabilize covariance matrix."""
    try:
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, min_eig)
        
        if np.max(eigvals) / np.min(eigvals) > max_cond:
            eigvals = np.maximum(eigvals, np.max(eigvals) / max_cond)
        
        return eigvecs @ np.diag(eigvals) @ eigvecs.T
    except:
        return np.eye(cov.shape[0], dtype=DTYPE) * min_eig

def log_gaussian(x: np.ndarray, mean: np.ndarray, cov: np.ndarray) -> float:
    """Log Gaussian density."""
    try:
        L = np.linalg.cholesky(cov)
        diff = x - mean
        alpha = np.linalg.solve(L, diff)
        logdet = 2.0 * float(np.sum(np.log(np.diag(L))))
        quad = float(alpha.T @ alpha)
        return float(-0.5 * (quad + logdet + len(x) * np.log(2.0 * np.pi)))
    except:
        return -1e10

# Note: laplace_evidence is unused in the current workflow (diagnostics computed inline)


def log_student_t(x: np.ndarray, mu: np.ndarray, Sigma: np.ndarray, nu: float = 4.0) -> float:
    """Log density of multivariate Student-t distribution."""
    d = len(x)
    delta = x - mu
    try:
        Sigma_inv = np.linalg.inv(Sigma)
        mahal = float(delta.T @ Sigma_inv @ delta)
        logdet = float(np.log(np.linalg.det(Sigma)))
    except np.linalg.LinAlgError:
        return -np.inf
    
    from scipy.special import gammaln
    log_norm = (gammaln((nu + d) / 2.0) - gammaln(nu / 2.0) - 
                0.5 * d * np.log(nu * np.pi) - 0.5 * logdet)
    log_kernel = -(nu + d) / 2.0 * np.log(1.0 + mahal / nu)
    return float(log_norm + log_kernel)

def compute_pe_given_cov(y: np.ndarray, omega: float, psi_map: np.ndarray, Sigma: np.ndarray, dt: float,
                         q_var: float, r_var: float, base_N: int,
                         proposal_scales: list[float], rng_seed: int = 0) -> tuple[float, float, float, float, np.ndarray, np.ndarray]:
    """Compute PE with raw IS given a fixed covariance Sigma (no Hessian recompute).
    Uses mixture proposal: 70% Gaussian (scale s) + 30% Gaussian (tail, scale tail_factor*s).
    This keeps sampling and density consistent and improves tail coverage.
    """
    rng = np.random.default_rng(rng_seed)

    best_pe = -np.inf
    best_ess_rel = 0.0
    best_scale = 1.0
    best_samples = None
    best_weights = None
    tail_factor = 6.0

    for scale_factor in proposal_scales:
        cov_main = Sigma * scale_factor
        cov_tail = Sigma * (scale_factor * tail_factor)

        # Mixture sampling
        n_main = int(0.7 * base_N)
        n_tail = base_N - n_main

        samples_main = rng.multivariate_normal(psi_map, cov_main, size=n_main).astype(DTYPE)
        samples_tail = rng.multivariate_normal(psi_map, cov_tail, size=n_tail).astype(DTYPE)
        samples = np.vstack([samples_main, samples_tail])
        rng.shuffle(samples)

        # Clip samples to bounds
        samples[:, 0] = np.clip(samples[:, 0], CFG.bounds[0][0], CFG.bounds[0][1])
        samples[:, 1] = np.clip(samples[:, 1], CFG.bounds[1][0], CFG.bounds[1][1])

        # Compute log proposal density for mixture q(x) = 0.7*N_main + 0.3*N_tail
        log_q_main = np.array([log_gaussian(s, psi_map, cov_main) for s in samples], dtype=DTYPE)
        log_q_tail = np.array([log_gaussian(s, psi_map, cov_tail) for s in samples], dtype=DTYPE)
        log_q = np.logaddexp(np.log(0.7) + log_q_main, np.log(0.3) + log_q_tail)

        # Compute log target density (log_joint)
        log_target = np.array([log_joint(y, omega, s, dt, q_var, r_var) for s in samples], dtype=DTYPE)

        # Compute log weights
        logw = log_target - log_q

        # Raw IS estimate (log-mean-exp)
        a = np.max(logw)
        pe = a + np.log(np.mean(np.exp(logw - a)))

        # ESS calculation for raw IS
        w = np.exp(logw - a)
        w_norm = w / np.sum(w)
        ess = 1.0 / np.sum(w_norm**2)
        ess_rel = ess / base_N

        if ess_rel > best_ess_rel:
            best_ess_rel = ess_rel
            best_pe = pe
            best_scale = scale_factor
            best_samples = samples
            best_weights = w_norm

    return float(best_pe), 0.9, float(best_ess_rel), float(best_scale), best_samples, best_weights

# Helper: draw IS samples and normalized weights at a given omega
def draw_is_samples_and_weights(y: np.ndarray, omega: float, psi_map: np.ndarray, Sigma: np.ndarray,
                                dt: float, q_var: float, r_var: float,
                                base_N: int, scale_factor: float, rng_seed: int = 0,
                                tail_factor: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(rng_seed)
    cov_main = Sigma * scale_factor
    cov_tail = Sigma * (scale_factor * tail_factor)
    n_main = int(0.7 * base_N)
    n_tail = base_N - n_main
    samples_main = rng.multivariate_normal(psi_map, cov_main, size=n_main).astype(DTYPE)
    samples_tail = rng.multivariate_normal(psi_map, cov_tail, size=n_tail).astype(DTYPE)
    samples = np.vstack([samples_main, samples_tail])
    rng.shuffle(samples)
    samples[:, 0] = np.clip(samples[:, 0], CFG.bounds[0][0], CFG.bounds[0][1])
    samples[:, 1] = np.clip(samples[:, 1], CFG.bounds[1][0], CFG.bounds[1][1])
    log_q_main = np.array([log_gaussian(s, psi_map, cov_main) for s in samples], dtype=DTYPE)
    log_q_tail = np.array([log_gaussian(s, psi_map, cov_tail) for s in samples], dtype=DTYPE)
    log_q = np.logaddexp(np.log(0.7) + log_q_main, np.log(0.3) + log_q_tail)
    log_target = np.array([log_joint(y, omega, s, dt, q_var, r_var) for s in samples], dtype=DTYPE)
    logw = log_target - log_q
    a = float(np.max(logw))
    w = np.exp(logw - a)
    w_norm = w / np.sum(w)
    return samples, w_norm

# Helper: covariance ellipse points for 2x2 Sigma
def covariance_ellipse_points(mu: np.ndarray, Sigma: np.ndarray, nsig: float = 2.0, num: int = 200) -> tuple[np.ndarray, np.ndarray]:
    try:
        vals, vecs = np.linalg.eigh(Sigma)
        vals = np.maximum(vals, 1e-12)
    except np.linalg.LinAlgError:
        vals = np.array([1e-6, 1e-6], dtype=DTYPE)
        vecs = np.eye(2, dtype=DTYPE)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angles = np.linspace(0.0, 2.0*np.pi, num)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=0)
    scales = nsig * np.sqrt(vals)
    ellipse = (vecs @ (circle * scales[:, None]))
    x = mu[0] + ellipse[0, :]
    y = mu[1] + ellipse[1, :]
    return x, y

def run_bulletproof(seed: int = 0, make_plots: bool = True, write_files: bool = True):
    print("=" * 80)
    print("PROFILE EVIDENCE vs PROFILE LIKELIHOOD (SMIB)")
    print("=" * 80)
    print("Starting simulation...")
    dt = CFG.dt
    T = CFG.T_analysis
    q_var = CFG.q_std**2
    r_var = CFG.r_std**2

    try:
        xs, ys = simulate(CFG.omega_true, CFG.gamma_true, CFG.zeta_true, T, dt, CFG.x0, CFG.q_std, CFG.r_std, seed)
        print(f"Simulation complete: T={T}, data shape={ys.shape}")
    except Exception as e:
        print(f"Simulation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Coarse grid to find peak (denser)
    print("Starting coarse grid search...")
    # K (stiffness) search band: broadened for SMIB
    coarse = np.linspace(1.0, 3.0, 41, dtype=DTYPE)
    pl_coarse = []
    print(f"Coarse grid: {len(coarse)} points from {coarse[0]:.3f} to {coarse[-1]:.3f}")
    for i, w in enumerate(coarse):
        print(f"Coarse [{i+1:02d}/{len(coarse)}] omega={w:.3f}", end=" ", flush=True)
        psi0 = CFG.prior_mean.copy()
        psi_mle, plv = find_mle(ys, float(w), psi0, dt, q_var, r_var)
        pl_coarse.append(plv)
    pl_coarse = np.array(pl_coarse)
    w_star = float(coarse[int(np.argmax(pl_coarse))])
    print(f"PL peak found at omega = {w_star:.3f}")

    # Refined grid around the peak (denser, ensure true omega coverage)
    # Ensure true omega is well within the refined grid
    grid_half_width = max(0.25, abs(w_star - CFG.omega_true) + 0.10)
    try:
        n_points = int(os.environ.get('REFINE_POINTS', '25'))
        n_points = max(3, n_points)
    except Exception:
        n_points = 25
    refine = np.linspace(w_star - grid_half_width, w_star + grid_half_width, n_points, dtype=DTYPE)
    
    baseN = 3000
    proposal_scales = [0.7, 1.0, 1.4, 2.0, 4.0, 8.0]

    pl_vals = []
    pe_vals = []
    k_hats = []  # Will be 0.9 for raw IS
    ess_rels = []
    best_scales = []
    # Compensation diagnostics
    laplace_vals = []
    comp_gains = []  # Laplace evidence minus PL
    ridge_logvols = []  # 0.5*(d log(2*pi) + logdetSigma)
    prior_mahal = []
    # For compensation visualization
    psi_mle_list = []
    psi_map_list = []
    Sigma_list = []
    hess_conds = []

    print(f"\nRefined analysis on {len(refine)} points:")
    psi_prev = CFG.prior_mean.copy()  # Warm-start initialization
    
    for i, w in enumerate(refine):
        print(f"  [{i+1:02d}/{len(refine)}] omega={float(w):.3f}")
        
        # Warm-start from previous solution (or prior for first point)
        psi0 = psi_prev.copy()
        
        # PL
        psi_mle, pll = find_mle(ys, float(w), psi0, dt, q_var, r_var)
        pl_vals.append(pll)
        psi_mle_list.append(psi_mle.copy())
        
        # MAP once
        psi_map, _ = find_map(ys, float(w), psi0, dt, q_var, r_var)
        psi_prev = psi_map.copy()  # Update warm-start for next iteration
        psi_map_list.append(psi_map.copy())

        # One Hessian/Sigma per omega
        f_log_joint = lambda p: log_joint(ys, float(w), p, dt, q_var, r_var)
        H = hessian_fd(f_log_joint, psi_map, h=5e-4)
        try:
            Sigma_raw = -np.linalg.inv(H)
        except np.linalg.LinAlgError:
            Sigma_raw = -np.linalg.inv(H + 1e-3*np.eye(2, dtype=DTYPE))
        Sigma = stabilize_cov(Sigma_raw, min_eig=1e-3, max_cond=1e4)
        Sigma_list.append(Sigma.copy())
        try:
            hess_cond = float(np.linalg.cond(-H))
        except Exception:
            hess_cond = np.nan
        hess_conds.append(hess_cond)

        # PE with fixed Sigma (no duplicate Hessian)
        pe, k_hat, ess, best_scale, samples, weights = compute_pe_given_cov(
            ys, float(w), psi_map, Sigma, dt, q_var, r_var, baseN, proposal_scales, rng_seed=42+i
        )
        
        # Local scale refinement if ESS is low
        if ess < 0.1:
            print(f"    Refining scales for low ESS={ess:.3f}...")
            refined_scales = [best_scale * f for f in [0.7, 1.0, 1.4, 2.0]]
            pe_refined, _, ess_refined, best_scale_refined, _, _ = compute_pe_given_cov(
                ys, float(w), psi_map, Sigma, dt, q_var, r_var, baseN, refined_scales, rng_seed=42+i+100
            )
            if ess_refined > ess:
                pe, ess, best_scale = pe_refined, ess_refined, best_scale_refined
                print(f"    Improved: ESS={ess:.3f}, scale={best_scale:.1f}")
        
        pe_vals.append(pe)
        k_hats.append(k_hat)
        ess_rels.append(ess)
        best_scales.append(best_scale)
        # Store samples/weights for direct evidence
        # Ensure samples and weights are always numpy arrays for consistent indexing
        samples = np.asarray(samples)
        weights = np.asarray(weights).ravel() # Ensure weights are 1D
        if samples.ndim != 2 or samples.shape[1] != 2 or weights.ndim != 1 or samples.shape[0] != weights.shape[0]:
            print(f"    WARNING: Malformed samples/weights for omega={float(w):.3f}. Skipping cloud plot for this point.")
            samples = np.array([]) # Empty array to prevent further errors
            weights = np.array([])
        
        if i == 0:
            all_is_samples = []
            all_is_weights = []
            max_weight_share = []
            weight_cv2 = []
        all_is_samples.append(samples)
        all_is_weights.append(weights)
        try:
            wnorm = np.asarray(weights).ravel()
            mw = float(np.max(wnorm)) if wnorm.size else np.nan
            mu = float(np.mean(wnorm)) if wnorm.size else np.nan
            var = float(np.var(wnorm)) if wnorm.size else np.nan
            cv2 = float(var / (mu * mu)) if (mu and mu != 0.0) else np.nan
        except Exception:
            mw, cv2 = np.nan, np.nan
        max_weight_share.append(mw)
        weight_cv2.append(cv2)
        
        # Laplace evidence and compensation metrics (reuse Sigma)
        try:
            Ls = np.linalg.cholesky(Sigma)
            logdet = 2.0 * float(np.sum(np.log(np.diag(Ls))))
        except np.linalg.LinAlgError:
            logdet = float(np.log(max(np.linalg.det(Sigma), 1e-24)))
        lj = log_joint(ys, float(w), psi_map, dt, q_var, r_var)
        d = len(psi_map)
        ridge_logvol = 0.5 * (d * np.log(2.0 * np.pi) + logdet)
        lap = float(lj + ridge_logvol)
        laplace_vals.append(lap)
        comp_gains.append(lap - pll)
        ridge_logvols.append(ridge_logvol)
        # Prior Mahalanobis distance at MAP
        try:
            prior_inv = np.linalg.inv(CFG.prior_cov)
            delta = psi_map - CFG.prior_mean
            prior_mahal.append(float(delta.T @ prior_inv @ delta))
        except np.linalg.LinAlgError:
            prior_mahal.append(np.nan)

        status = "OK" if (ess >= 0.1) else ("WARN" if ess >= 0.05 else "POOR")
        print(f"    {status:4s} PL={pll:8.2f} | PE={pe:8.2f} | ESS={ess:5.3f} (scale={best_scale:.1f})")

    pl_vals = np.array(pl_vals)
    pe_vals = np.array(pe_vals)
    k_hats = np.array(k_hats)
    ess_rels = np.array(ess_rels)
    laplace_vals = np.array(laplace_vals)
    comp_gains = np.array(comp_gains)
    ridge_logvols = np.array(ridge_logvols)
    prior_mahal = np.array(prior_mahal)
    psi_mle_arr = np.array(psi_mle_list)
    psi_map_arr = np.array(psi_map_list)
    all_is_samples = np.array(all_is_samples, dtype=object).tolist()
    all_is_weights = np.array(all_is_weights, dtype=object).tolist()

    # Per-curve normalization
    pl_norm = pl_vals - np.max(pl_vals)
    pe_norm = pe_vals - np.max(pe_vals)
    # Correction detector: positive where PE lifts support vs PL
    delta_norm = pe_norm - pl_norm

    # CI using common threshold
    thr = -0.5 * 3.841458820694124
    def ci_from_curve(grid, curve):
        idx = np.where(curve >= thr)[0]
        return (np.nan, np.nan) if idx.size == 0 else (float(grid[idx[0]]), float(grid[idx[-1]]))
    pl_ci = ci_from_curve(refine, pl_norm)
    pe_ci = ci_from_curve(refine, pe_norm)

    # Plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    ax1 = axes[0, 0]
    ax1.plot(refine, pl_norm, 'b-o', label='PL', linewidth=2, markersize=3)
    ax1.plot(refine, pe_norm, 'r-s', label='PE (Raw IS)', linewidth=2, markersize=3)
    ax1.axvline(CFG.omega_true, color='green', linestyle='--', label='True omega', alpha=0.8)
    ax1.axhline(thr, color='gray', linestyle=':', alpha=0.7, label='95% threshold')
    if not (np.isnan(pl_ci[0]) or np.isnan(pl_ci[1])):
        ax1.axvspan(pl_ci[0], pl_ci[1], color='blue', alpha=0.12)
    if not (np.isnan(pe_ci[0]) or np.isnan(pe_ci[1])):
        ax1.axvspan(pe_ci[0], pe_ci[1], color='red', alpha=0.12)
    ax1.set_title('Normalized Log-Likelihood/Evidence (Per-Curve Max)')
    ax1.set_xlabel('K (stiffness)')
    ax1.set_ylabel('Normalized log-scale')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    # Add zoom inset when curvature is small to make differences visible
    try:
        y_min = float(min(np.min(pl_norm), np.min(pe_norm)))
        y_range = abs(y_min)  # since max is 0 by normalization
        if y_range < 0.25:
            axins = inset_axes(ax1, width="42%", height="42%", loc="lower left", borderpad=1.2)
            axins.plot(refine, pl_norm, 'b-', linewidth=2)
            axins.plot(refine, pe_norm, 'r-', linewidth=2)
            axins.axhline(thr, color='gray', linestyle=':', alpha=0.7)
            axins.set_xlim(float(refine[0]), float(refine[-1]))
            axins.set_ylim(y_min - 0.02, 0.02)
            axins.set_title('Zoom', fontsize=9)
            axins.grid(True, alpha=0.3)
    except Exception:
        pass

    ax2 = axes[0, 1]
    ax2.plot(refine, ess_rels, 'darkgreen', marker='d')
    ax2.axhline(0.1, color='red', linestyle='--', alpha=0.7, label='ESS target (0.1)')
    ax2.axhline(0.05, color='orange', linestyle=':', alpha=0.7, label='ESS acceptable (0.05)')
    ax2.set_title('Relative ESS per omega (Raw IS)')
    ax2.set_xlabel('K (stiffness)'); ax2.set_ylabel('ESS'); ax2.legend(); ax2.grid(True, alpha=0.3)

    ax3 = axes[1, 0]
    ax3.plot(refine, best_scales, 'purple', marker='o')
    ax3.set_title('Optimal Proposal Scale Factor per omega')
    ax3.set_xlabel('K (stiffness)'); ax3.set_ylabel('Scale Factor'); ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 1]
    ax4.plot(refine, pl_vals, 'b-', alpha=0.7, label='PL (raw)')
    ax4_t = ax4.twinx()
    ax4_t.plot(refine, pe_vals, 'r-', alpha=0.7, label='PE (raw)')
    ax4.set_title('Raw Values (different scales)')
    ax4.set_xlabel('K (stiffness)'); ax4.set_ylabel('PL'); ax4_t.set_ylabel('PE')
    ax4.grid(True, alpha=0.3)

    if make_plots:
        plt.tight_layout()
        out_name = f"{get_run_prefix()}pe_results.png"
        plt.savefig(out_name, dpi=160, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot: {out_name}")

    if make_plots:
        # Compensation visualization
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 9))
        axc1 = axes2[0, 0]
        axc1.plot(refine, comp_gains, 'm-o')
        axc1.axhline(0.0, color='gray', linestyle=':')
        axc1.set_title('Compensation Gain: Laplace - PL (log-scale)')
        axc1.set_xlabel('K (stiffness)'); axc1.set_ylabel('Gain')
        axc1.grid(True, alpha=0.3)

        axc2 = axes2[0, 1]
        axc2.plot(refine, ridge_logvols, 'c-s')
        axc2.set_title('Ridge Log-Volume (0.5[d log 2*pi + logdetSigma])')
        axc2.set_xlabel('K (stiffness)'); axc2.set_ylabel('Log-Volume')
        axc2.grid(True, alpha=0.3)

        axc3 = axes2[1, 0]
        axc3.plot(refine, prior_mahal, 'k-d')
        axc3.set_title('Prior Mahalanobis Distance at MAP')
        axc3.set_xlabel('K (stiffness)'); axc3.set_ylabel('Mahalanobis^2')
        axc3.grid(True, alpha=0.3)

        axc4 = axes2[1, 1]
        axc4.scatter(ridge_logvols, comp_gains, c=refine, cmap='viridis', s=50)
        axc4.set_title('Gain vs Ridge Volume (color: omega)')
        axc4.set_xlabel('Ridge Log-Volume'); axc4.set_ylabel('Compensation Gain')
        axc4.grid(True, alpha=0.3)
        plt.tight_layout()
        comp_fig = f"{get_run_prefix()}pe_compensation.png"
        plt.savefig(comp_fig, dpi=160, bbox_inches='tight')
        plt.close(fig2)
        print(f"Saved compensation diagnostics: {comp_fig}")

        # Correction delta plot (PE_norm - PL_norm)
        fig3, axd = plt.subplots(figsize=(8, 4))
        axd.plot(refine, delta_norm, 'b-o')
        axd.axhline(0.0, color='gray', linestyle=':')
        axd.set_title('Correction Delta (PE_norm - PL_norm)')
        axd.set_xlabel('K (stiffness)'); axd.set_ylabel('Delta (PE - PL)')
        axd.grid(True, alpha=0.3)
        plt.tight_layout()
        delta_fig = f"{get_run_prefix()}pl_pe_delta.png"
        plt.savefig(delta_fig, dpi=160, bbox_inches='tight')
        plt.close(fig3)
        print(f"Saved correction delta plot: {delta_fig}")

        # Compensation path plots (how nuisance params move as omega changes)
        try:
            figp, axp = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
            axp[0].plot(refine, psi_mle_arr[:, 0], 'b-o', label='MLE gamma', ms=3)
            axp[0].plot(refine, psi_map_arr[:, 0], 'r-s', label='MAP gamma', ms=3, alpha=0.8)
            axp[0].axvline(CFG.omega_true, color='green', ls='--', alpha=0.7)
            axp[0].set_ylabel('gamma')
            axp[0].grid(True, alpha=0.3)
            axp[0].legend()
            axp[1].plot(refine, psi_mle_arr[:, 1], 'b-o', label='MLE zeta', ms=3)
            axp[1].plot(refine, psi_map_arr[:, 1], 'r-s', label='MAP zeta', ms=3, alpha=0.8)
            axp[1].axvline(CFG.omega_true, color='green', ls='--', alpha=0.7)
            axp[1].set_xlabel('K (stiffness)')
            axp[1].set_ylabel('zeta')
            axp[1].grid(True, alpha=0.3)
            plt.tight_layout()
            path_fig = f"{get_run_prefix()}compensation_path.png"
            plt.savefig(path_fig, dpi=160, bbox_inches='tight')
            plt.close(figp)
            print(f"Saved compensation path: {path_fig}")
        except Exception as e:
            print(f"Compensation path plot failed: {e}")

        # Standardize colorbar across clouds within this run
        try:
            global_wmax = 0.0
            for wts in all_is_weights:
                if wts is None:
                    continue
                ww = np.asarray(wts).ravel()
                if ww.size == 0:
                    continue
                global_wmax = max(global_wmax, float(np.max(ww)))
            if not np.isfinite(global_wmax) or global_wmax <= 0.0:
                global_wmax = 1.0
        except Exception:
            global_wmax = 1.0
        if os.environ.get('CLOUD_VMAX', '').lower() in ('1', 'true', 'unit', 'fixed'):
            global_wmax = 1.0

        # Compensation cloud plots at high positive-delta omega points (PE > PL)
        high_pos_delta_omegas = refine[np.where(delta_norm > 0.05)[0]]
        for w_idx, w_val in enumerate(refine):
            if w_val in high_pos_delta_omegas:
                samples = np.asarray(all_is_samples[w_idx])
                weights = np.asarray(all_is_weights[w_idx]).ravel()
                if samples.ndim != 2 or samples.shape[1] < 2:
                    continue
                psi_map_at_w = psi_map_arr[w_idx]
                Sigma_w = Sigma_list[w_idx]
                ess_val = float(ess_rels[w_idx])
                status = 'OK' if ess_val >= 0.1 else ('WARN' if ess_val >= 0.05 else 'POOR')
                scale_val = float(best_scales[w_idx])
                delta_val = float(delta_norm[w_idx])

                figc, axc = plt.subplots(figsize=(8, 6))
                sc = axc.scatter(samples[:, 0], samples[:, 1], c=weights, cmap='viridis', vmin=0.0, vmax=global_wmax, s=50, alpha=0.7)
                # 2σ Laplace ellipse around MAP using Sigma + major-axis arrow
                try:
                    ex, ey = covariance_ellipse_points(psi_map_at_w, Sigma_w, nsig=2.0)
                    axc.plot(ex, ey, 'k--', linewidth=1.5, label='2σ ellipse')
                    vals, vecs = np.linalg.eigh(Sigma_w)
                    order = np.argsort(vals)[::-1]
                    v1 = vecs[:, order[0]]
                    scale_arrow = 2.0 * np.sqrt(max(vals[order[0]], 1e-12))
                    axc.arrow(psi_map_at_w[0], psi_map_at_w[1], v1[0]*scale_arrow, v1[1]*scale_arrow,
                              head_width=0.02, head_length=0.02, fc='k', ec='k', alpha=0.8)
                    theta = float(np.degrees(np.arctan2(v1[1], v1[0])))
                except Exception:
                    theta = np.nan
                axc.plot(psi_map_at_w[0], psi_map_at_w[1], 'rx', markersize=10, label='MAP')
                axc.set_xlabel('gamma')
                axc.set_ylabel('zeta')
                title_extra = ''
                try:
                    title_extra = f"; maxw={float(np.max(weights)):.3f}; CV2={float(np.var(weights)/(np.mean(weights)**2)):.1f}"
                except Exception:
                    pass
                title_theta = f"; θ={theta:.1f}°" if np.isfinite(theta) else ''
                axc.set_title(f'IS at K={w_val:.3f} (delta={delta_val:+.3f}; ESS={ess_val:.3f} {status}; N={baseN}; scale={scale_val:.1f}{title_theta}{title_extra})')
                axc.legend()
                axc.grid(True, alpha=0.3)
                plt.colorbar(sc, ax=axc, label='Normalized weight')
                plt.tight_layout()
                comp_cloud_fig = f"{get_run_prefix()}compensation_cloud_omega_{w_val:.3f}.png"
                plt.savefig(comp_cloud_fig, dpi=160, bbox_inches='tight')
                plt.close(figc)
                print(f"Saved compensation cloud plot: {comp_cloud_fig}")

        # Overestimation cloud plots at high negative-delta omega points (PE < PL)
        high_neg_delta_omegas = refine[np.where(delta_norm < -0.05)[0]]
        for w_idx, w_val in enumerate(refine):
            if w_val in high_neg_delta_omegas:
                samples = np.asarray(all_is_samples[w_idx])
                weights = np.asarray(all_is_weights[w_idx]).ravel()
                if samples.ndim != 2 or samples.shape[1] < 2:
                    continue
                psi_map_at_w = psi_map_arr[w_idx]
                Sigma_w = Sigma_list[w_idx]
                ess_val = float(ess_rels[w_idx])
                status = 'OK' if ess_val >= 0.1 else ('WARN' if ess_val >= 0.05 else 'POOR')
                scale_val = float(best_scales[w_idx])
                delta_val = float(delta_norm[w_idx])

                figc2, axc2 = plt.subplots(figsize=(8, 6))
                sc2 = axc2.scatter(samples[:, 0], samples[:, 1], c=weights, cmap='plasma', vmin=0.0, vmax=global_wmax, s=50, alpha=0.7)
                try:
                    ex2, ey2 = covariance_ellipse_points(psi_map_at_w, Sigma_w, nsig=2.0)
                    axc2.plot(ex2, ey2, 'k--', linewidth=1.5, label='2σ ellipse')
                    vals2, vecs2 = np.linalg.eigh(Sigma_w)
                    order2 = np.argsort(vals2)[::-1]
                    v1_2 = vecs2[:, order2[0]]
                    scale_arrow2 = 2.0 * np.sqrt(max(vals2[order2[0]], 1e-12))
                    axc2.arrow(psi_map_at_w[0], psi_map_at_w[1], v1_2[0]*scale_arrow2, v1_2[1]*scale_arrow2,
                               head_width=0.02, head_length=0.02, fc='k', ec='k', alpha=0.8)
                    theta2 = float(np.degrees(np.arctan2(v1_2[1], v1_2[0])))
                except Exception:
                    theta2 = np.nan
                axc2.plot(psi_map_at_w[0], psi_map_at_w[1], 'rx', markersize=10, label='MAP')
                axc2.set_xlabel('gamma')
                axc2.set_ylabel('zeta')
                title_extra2 = ''
                try:
                    title_extra2 = f"; maxw={float(np.max(weights)):.3f}; CV2={float(np.var(weights)/(np.mean(weights)**2)):.1f}"
                except Exception:
                    pass
                title_theta2 = f"; θ={theta2:.1f}°" if np.isfinite(theta2) else ''
                axc2.set_title(f'IS at K={w_val:.3f} (delta={delta_val:+.3f}; ESS={ess_val:.3f} {status}; N={baseN}; scale={scale_val:.1f}{title_theta2}{title_extra2})')
                axc2.legend()
                axc2.grid(True, alpha=0.3)
                plt.colorbar(sc2, ax=axc2, label='Normalized weight')
                plt.tight_layout()
                over_cloud_fig = f"{get_run_prefix()}overestimation_cloud_omega_{w_val:.3f}.png"
                plt.savefig(over_cloud_fig, dpi=160, bbox_inches='tight')
                plt.close(figc2)
                print(f"Saved overestimation cloud plot: {over_cloud_fig}")

        # Predictive checks: RMSE vs omega, plus QQ and ACF at peak PL
        try:
            def compute_innovations(y_in: np.ndarray, k_in: float, psi_in: np.ndarray) -> np.ndarray:
                Tloc = y_in.shape[0]
                A = A_matrix(k_in, float(psi_in[0]), dt, float(psi_in[1]), CFG.M_inertia)
                Q = q_var * np.eye(2, dtype=DTYPE)
                R = np.array([[r_var, 0.0],[0.0, (CFG.r_std2**2)]], dtype=DTYPE)
                x = np.zeros(2, dtype=DTYPE)
                P = 0.2 * np.eye(2, dtype=DTYPE)
                innovs = []
                for t in range(Tloc):
                    x_pred = A @ x
                    P_pred = A @ P @ A.T + Q
                    S = P_pred + R
                    try:
                        L = np.linalg.cholesky(S)
                        K = P_pred @ np.linalg.solve(S, np.eye(2, dtype=DTYPE))
                        innov = y_in[t] - x_pred
                        x = x_pred + K @ innov
                        P = (np.eye(2, dtype=DTYPE) - K) @ P_pred
                        innovs.append(innov)
                    except np.linalg.LinAlgError:
                        innovs.append(np.array([np.nan, np.nan], dtype=DTYPE))
                return np.array(innovs)

            rmse_curve = []
            for w_idx, w_val in enumerate(refine):
                innovs = compute_innovations(ys, float(w_val), psi_map_arr[w_idx])
                rmse_curve.append(float(np.sqrt(np.nanmean(np.sum(innovs**2, axis=1)))))
            rmse_curve = np.array(rmse_curve)

            # QQ and ACF at PL peak
            peak_idx = int(np.argmax(pl_vals))
            innovs_peak = compute_innovations(ys, float(refine[peak_idx]), psi_map_arr[peak_idx])
            resid = innovs_peak[:, 0]
            resid = resid[np.isfinite(resid)]
            resid_sorted = np.sort(resid)
            n = resid_sorted.size
            if n > 0:
                q_theory = np.sort(np.random.normal(size=n))
                # ACF up to lag 40
                maxlag = min(40, n-2)
                resid_centered = resid - np.nanmean(resid)
                acf = [1.0]
                denom = float(np.nanvar(resid_centered)) * n if np.nanvar(resid_centered) > 0 else 1.0
                for lag in range(1, maxlag+1):
                    acf.append(float(np.nansum(resid_centered[:-lag] * resid_centered[lag:]) / denom))

                figpchk, axs = plt.subplots(1, 3, figsize=(16, 4))
                axs[0].plot(refine, rmse_curve, 'k-o')
                axs[0].set_title('RMSE of innovations vs K')
                axs[0].set_xlabel('K (stiffness)'); axs[0].set_ylabel('RMSE')
                axs[1].plot(q_theory, resid_sorted, 'bo', alpha=0.6)
                axs[1].plot([q_theory.min(), q_theory.max()], [q_theory.min(), q_theory.max()], 'r--')
                axs[1].set_title('QQ plot (innovations, ch1) at PL peak')
                axs[1].set_xlabel('Theoretical quantiles'); axs[1].set_ylabel('Sample quantiles')
                axs[2].stem(range(len(acf)), acf)
                axs[2].set_title('ACF of innovations (ch1) at PL peak')
                axs[2].set_xlabel('lag'); axs[2].set_ylabel('ACF')
                plt.tight_layout()
                pred_fig = f"{get_run_prefix()}predictive_checks.png"
                plt.savefig(pred_fig, dpi=160, bbox_inches='tight')
                plt.close(figpchk)
                print(f"Saved predictive checks: {pred_fig}")
        except Exception as e:
            print(f"Predictive checks failed: {e}")
    
    if CFG.enable_comp_demo:
        print("Compensation demo ENABLED: zeta jitter was injected to create PE>PL opportunities.")
    else:
        print("Compensation demo DISABLED: using stationary nuisance amplitude.")

        # Save a concise compensation summary
    if write_files:
        try:
            top_idx = int(np.argmax(comp_gains))
            comp_summary_path = f"{get_run_prefix()}compensation_summary.txt"
            with open(comp_summary_path, 'w', encoding='utf-8') as fh:
                fh.write(f"Peak compensation at omega={refine[top_idx]:.3f}: gain={comp_gains[top_idx]:.3f}, ridge_logvol={ridge_logvols[top_idx]:.3f}, prior_mahal={prior_mahal[top_idx]:.3f}\n")
                fh.write(f"Mean gain={np.mean(comp_gains):.3f}, fraction gain>0: {float(np.mean(comp_gains>0)):.2f}\n")
            print(f"Saved compensation summary: {comp_summary_path}")
        except Exception as e:
            print(f"Compensation summary save failed: {e}")

        # Save correction and overestimation flags summary
    if write_files:
        try:
            corrected_idxs = np.where(delta_norm > 0.05)[0]
            correction_flags_path = f"{get_run_prefix()}correction_flags.txt"
            with open(correction_flags_path, 'w', encoding='utf-8') as fh:
                fh.write('omega where PE_norm - PL_norm > 0.05 (PE corrects PL optimism):\n')
                for idx in corrected_idxs:
                    fh.write(f"  omega={refine[idx]:.3f}, delta={delta_norm[idx]:+.3f}, ridge={ridge_logvols[idx]:.3f}\n")
                fh.write(f"Total corrected points: {len(corrected_idxs)}/{len(refine)}\n")
            print(f"Saved correction flags: {correction_flags_path}")
        except Exception as e:
            print(f"Correction flags save failed: {e}")
        try:
            over_idx = np.where(delta_norm < -0.05)[0]
            overestimation_flags_path = f"{get_run_prefix()}overestimation_flags.txt"
            with open(overestimation_flags_path, 'w', encoding='utf-8') as fh:
                fh.write('omega where PE_norm - PL_norm < -0.05 (PL overestimation reduced by PE):\n')
                for idx in over_idx:
                    fh.write(f"  omega={refine[idx]:.3f}, delta={delta_norm[idx]:+.3f}, ridge={ridge_logvols[idx]:.3f}\n")
                fh.write(f"Total overestimation points: {len(over_idx)}/{len(refine)}\n")
            print(f"Saved overestimation flags: {overestimation_flags_path}")
        except Exception as e:
            print(f"Overestimation flags save failed: {e}")

        # Optional: Peak overestimation summary
        try:
            if np.any(delta_norm < -0.05):
                neg_idx = int(np.argmin(delta_norm))
                overestimation_summary_path = f"{get_run_prefix()}overestimation_summary.txt"
                with open(overestimation_summary_path, 'w', encoding='utf-8') as fh:
                    fh.write(
                        f"Peak overestimation at omega={refine[neg_idx]:.3f}: "
                        f"delta={delta_norm[neg_idx]:+.3f}, ridge_logvol={ridge_logvols[neg_idx]:.3f}, "
                        f"ESS={ess_rels[neg_idx]:.3f}\n"
                    )
                    fh.write(f"Total points with delta<-0.05: {int(np.sum(delta_norm < -0.05))}/{len(refine)}\n")
                print(f"Saved overestimation summary: {overestimation_summary_path}")
        except Exception as e:
            print(f"Overestimation summary save failed: {e}")

    # Summary
    ess_ok_rate = float(np.mean(ess_rels >= 0.1))
    ess_acceptable_rate = float(np.mean(ess_rels >= 0.05))
    print("-"*80)
    print(f"ESS quality (>=0.1): {100.0*ess_ok_rate:.1f}% | ESS acceptable (>=0.05): {100.0*ess_acceptable_rate:.1f}%")
    
    pl_w = np.nan
    pe_w = np.nan
    if not (np.isnan(pl_ci[0]) or np.isnan(pl_ci[1])):
        pl_w = pl_ci[1] - pl_ci[0]
        print(f"PL 95% CI: {pl_ci} (width={pl_w:.3f})")
    if not (np.isnan(pe_ci[0]) or np.isnan(pe_ci[1])):
        pe_w = pe_ci[1] - pe_ci[0]
        print(f"PE 95% CI: {pe_ci} (width={pe_w:.3f})")
    
    if not np.isnan(pl_w) and not np.isnan(pe_w):
        if pe_w > pl_w * 1.1:
            print("RESULT: PE is more conservative than PL (fair)")
        else:
            print("RESULT: PE and PL have similar widths (fair)")
    
    print("PROFILE EVIDENCE ANALYSIS COMPLETE!")
    
    # Save results for later analysis
    results = {
        'omegas': refine,
        'pl_vals': pl_vals,
        'pe_vals': pe_vals,
        'laplace_vals': laplace_vals,
        'comp_gains': comp_gains,
        'ridge_logvols': ridge_logvols,
        'prior_mahal': prior_mahal,
        'ess_rels': ess_rels,
        'best_scales': best_scales,
        'hess_conds': np.array(hess_conds),
        'pl_ci': pl_ci,
        'pe_ci': pe_ci,
        'pl_ci_width': pl_w,
        'pe_ci_width': pe_w,
        'ess_ok_rate': ess_ok_rate,
        'ess_acceptable_rate': ess_acceptable_rate,
        'omega_true': CFG.omega_true,
        'delta_norm': delta_norm,
        # Direct-evidence support arrays for offline rendering
        'psi_mle_arr': psi_mle_arr,
        'psi_map_arr': psi_map_arr,
        'all_is_samples': all_is_samples,
        'all_is_weights': all_is_weights,
        'max_weight_share': np.array(max_weight_share) if 'max_weight_share' in locals() else None,
        'weight_cv2': np.array(weight_cv2) if 'weight_cv2' in locals() else None
    }
    if write_files:
        results_pkl_path = f"{get_run_prefix()}pe_results.pkl"
        with open(results_pkl_path, 'wb') as f:
            pickle.dump(results, f)
        # Write brief summary file with objective indicators
        try:
            true_idx = int(np.argmin(np.abs(refine - CFG.omega_true)))
            summary_txt_path = f"{get_run_prefix()}pe_vs_pl_summary.txt"
            with open(summary_txt_path,'w', encoding='utf-8') as fh:
                fh.write(f"True omega={CFG.omega_true:.3f}, PL@true={pl_vals[true_idx]:.3f}, PE@true={pe_vals[true_idx]:.3f}, Gain@true={float(laplace_vals[true_idx]-pl_vals[true_idx]):.3f}\n")
                fh.write(f"Fraction omega with PE>PL (Laplace gain>0): {float(np.mean(comp_gains>0)):.2f}\n")
                fh.write(f"ESS mean={float(np.mean(ess_rels)):.3f}, std={float(np.std(ess_rels)):.3f}\n")
        except Exception as e:
            print(f"Summary write failed: {e}")
        print(f"Saved results to {results_pkl_path}")

        # Write manifest for reproducibility
        try:
            import sys, platform, subprocess, json as _json
            manifest = {
                'model': os.environ.get('MODEL_NAME', 'smib'),
                'seed': int(seed),
                'config': {
                    'enable_comp_demo': bool(CFG.enable_comp_demo),
                    'zeta_noise_dist': str(CFG.zeta_noise_dist)
                },
                'timestamp': __import__('datetime').datetime.now().isoformat(),
                'python_version': sys.version,
                'platform': platform.platform(),
                'numpy_version': __import__('numpy').__version__,
                'scipy_version': __import__('scipy').__version__
            }
            try:
                gh = subprocess.run(['git', 'rev-parse', 'HEAD'], capture_output=True, text=True)
                if gh.returncode == 0:
                    manifest['git_commit'] = gh.stdout.strip()
            except Exception:
                pass
            man_path = f"{get_run_prefix()}manifest.json"
            with open(man_path, 'w', encoding='utf-8') as mf:
                _json.dump(manifest, mf, indent=2)
            print(f"Saved manifest: {man_path}")
        except Exception as e:
            print(f"Manifest write failed: {e}")

    # Return results for programmatic use
    results['delta_norm'] = delta_norm
    return results

if __name__ == '__main__':
    run_bulletproof()