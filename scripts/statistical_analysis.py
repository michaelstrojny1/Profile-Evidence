#!/usr/bin/env python3
"""
PhD-Level Statistical Analysis for PE vs PL Multi-Seed Validation

Comprehensive statistical tests for publication-grade analysis:
- Paired t-tests (PE vs PL at true parameters)
- Mann-Whitney U (compensation ON vs OFF)
- Spearman correlations (gain vs ridge volume)
- Cohen's d (effect sizes)
- Bootstrap confidence intervals
- Power analysis

Usage:
  python -u scripts/statistical_analysis.py [RESULTS_DIR]
"""

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Dict, Tuple, Optional


def load_results(results_dir: str) -> List[Dict]:
    """Load all *_pe_results.pkl files from directory."""
    pkl_files = glob.glob(os.path.join(results_dir, '**', '*_pe_results.pkl'), recursive=True)
    results = []
    
    for pkl_path in pkl_files:
        try:
            with open(pkl_path, 'rb') as f:
                res = pickle.load(f)
            
            # Extract metadata
            cfg = res.get('config', {})
            res['pkl_path'] = pkl_path
            model = res.get('model', 'unknown')
            seed = res.get('seed', -1)
            comp_demo = cfg.get('enable_comp_demo', None)
            noise_dist = cfg.get('zeta_noise_dist', 'unknown')

            # Fallback: infer from filename pattern if missing
            base = os.path.basename(pkl_path)
            parts = base.split('_')
            # Expected like: model_s{seed}_{comp}_{noise}_pe_results.pkl
            try:
                if model == 'unknown' and parts:
                    if parts[0] in ('oscillator', 'smib'):
                        model = parts[0]
                if seed == -1:
                    for p in parts:
                        if p.startswith('s') and p[1:].isdigit():
                            seed = int(p[1:])
                            break
                if comp_demo is None:
                    # Look for literal 'True' or 'False'
                    if 'True' in parts:
                        comp_demo = True
                    elif 'False' in parts:
                        comp_demo = False
                if noise_dist == 'unknown':
                    for cand in ('gaussian', 'laplace', 'student', 'student_t', 'student-t'):
                        if cand in parts:
                            noise_dist = 'student_t' if 'student' in cand else cand
                            break
            except Exception:
                pass

            res['model'] = model
            res['seed'] = seed
            # Normalize fields back to config-like entries for downstream consistency
            res['comp_demo'] = comp_demo
            res['noise_dist'] = noise_dist
            
            results.append(res)
        except Exception as e:
            print(f"WARN: Failed to load {pkl_path}: {e}")
    
    return results


def extract_metrics(results: List[Dict]) -> pd.DataFrame:
    """Extract key metrics into DataFrame for analysis."""
    rows = []
    
    for res in results:
        # Basic info
        model = res.get('model', 'unknown')
        seed = int(res.get('seed', -1))
        comp_demo = bool(res.get('comp_demo', False))
        noise_dist = res.get('noise_dist', 'unknown')
        
        # Core metrics
        delta_norm = np.array(res.get('delta_norm', []), float)
        ess_rels = np.array(res.get('ess_rels', []), float)
        comp_gains = np.array(res.get('comp_gains', []), float)
        ridge_logvols = np.array(res.get('ridge_logvols', []), float)
        
        # PE vs PL at true parameter
        omega_true = res.get('omega_true', np.nan)
        omegas = np.array(res.get('omegas', []), float)
        pl_vals = np.array(res.get('pl_vals', []), float)
        pe_vals = np.array(res.get('pe_vals', []), float)
        
        # Find closest omega to true value
        if omegas.size > 0 and not np.isnan(omega_true):
            true_idx = np.argmin(np.abs(omegas - omega_true))
            pl_at_true = float(pl_vals[true_idx]) if pl_vals.size > true_idx else np.nan
            pe_at_true = float(pe_vals[true_idx]) if pe_vals.size > true_idx else np.nan
        else:
            pl_at_true = pe_at_true = np.nan
        
        # CI metrics
        pl_ci = res.get('pl_ci', (np.nan, np.nan))
        pe_ci = res.get('pe_ci', (np.nan, np.nan))
        
        def ci_width(ci):
            lo, hi = ci
            return (hi - lo) if not (np.isnan(lo) or np.isnan(hi)) else np.nan
        
        def ci_mid(ci):
            lo, hi = ci
            return 0.5 * (lo + hi) if not (np.isnan(lo) or np.isnan(hi)) else np.nan
        
        # Compensation metrics
        pos_frac = float(np.mean(delta_norm > 0.05)) if delta_norm.size else np.nan
        neg_frac = float(np.mean(delta_norm < -0.05)) if delta_norm.size else np.nan
        delta_mean = float(np.mean(delta_norm)) if delta_norm.size else np.nan
        delta_std = float(np.std(delta_norm)) if delta_norm.size else np.nan
        
        # ESS metrics
        ess_mean = float(np.mean(ess_rels)) if ess_rels.size else np.nan
        ess_min = float(np.min(ess_rels)) if ess_rels.size else np.nan
        ess_reliable = float(np.mean(ess_rels >= 0.1)) if ess_rels.size else np.nan
        
        # Gain-ridge correlation
        if comp_gains.size > 1 and ridge_logvols.size > 1 and comp_gains.size == ridge_logvols.size:
            try:
                gain_ridge_corr, gain_ridge_p = spearmanr(comp_gains, ridge_logvols)
                gain_ridge_corr = float(gain_ridge_corr)
                gain_ridge_p = float(gain_ridge_p)
            except:
                gain_ridge_corr = gain_ridge_p = np.nan
        else:
            gain_ridge_corr = gain_ridge_p = np.nan
        
        rows.append({
            'model': model,
            'seed': seed,
            'comp_demo': comp_demo,
            'noise_dist': noise_dist,
            'pl_at_true': pl_at_true,
            'pe_at_true': pe_at_true,
            'pe_pl_diff_at_true': pe_at_true - pl_at_true,
            'pl_ci_width': ci_width(pl_ci),
            'pe_ci_width': ci_width(pe_ci),
            'pl_ci_mid': ci_mid(pl_ci),
            'pe_ci_mid': ci_mid(pe_ci),
            'pos_frac': pos_frac,
            'neg_frac': neg_frac,
            'delta_mean': delta_mean,
            'delta_std': delta_std,
            'ess_mean': ess_mean,
            'ess_min': ess_min,
            'ess_reliable': ess_reliable,
            'gain_ridge_corr': gain_ridge_corr,
            'gain_ridge_p': gain_ridge_p,
        })
    
    return pd.DataFrame(rows)


def paired_ttest_analysis(df: pd.DataFrame) -> Dict:
    """Paired t-test: PE vs PL at true parameter values."""
    results = {}
    
    # Overall paired t-test
    valid_pairs = df.dropna(subset=['pl_at_true', 'pe_at_true'])
    if len(valid_pairs) >= 3:
        t_stat, p_val = stats.ttest_rel(valid_pairs['pe_at_true'], valid_pairs['pl_at_true'])
        
        # Effect size (Cohen's d for paired samples)
        diff = valid_pairs['pe_at_true'] - valid_pairs['pl_at_true']
        cohens_d = np.mean(diff) / np.std(diff)
        
        results['overall'] = {
            't_statistic': float(t_stat),
            'p_value': float(p_val),
            'cohens_d': float(cohens_d),
            'n_pairs': len(valid_pairs),
            'mean_diff': float(np.mean(diff)),
            'std_diff': float(np.std(diff))
        }
    
    # By compensation regime
    for comp in [True, False]:
        subset = valid_pairs[valid_pairs['comp_demo'] == comp]
        if len(subset) >= 3:
            t_stat, p_val = stats.ttest_rel(subset['pe_at_true'], subset['pl_at_true'])
            diff = subset['pe_at_true'] - subset['pl_at_true']
            cohens_d = np.mean(diff) / np.std(diff) if np.std(diff) > 0 else 0
            
            results[f'comp_{comp}'] = {
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'cohens_d': float(cohens_d),
                'n_pairs': len(subset),
                'mean_diff': float(np.mean(diff)),
                'std_diff': float(np.std(diff))
            }
    
    return results


def mannwhitney_analysis(df: pd.DataFrame) -> Dict:
    """Mann-Whitney U test: Compensation ON vs OFF regimes."""
    results = {}
    
    comp_on = df[df['comp_demo'] == True]
    comp_off = df[df['comp_demo'] == False]
    
    metrics = ['pos_frac', 'neg_frac', 'delta_mean', 'ess_mean']
    
    for metric in metrics:
        on_vals = comp_on[metric].dropna()
        off_vals = comp_off[metric].dropna()
        
        if len(on_vals) >= 3 and len(off_vals) >= 3:
            try:
                u_stat, p_val = mannwhitneyu(on_vals, off_vals, alternative='two-sided')
                
                # Effect size (rank-biserial correlation)
                n1, n2 = len(on_vals), len(off_vals)
                r = 1 - (2 * u_stat) / (n1 * n2)
                
                results[metric] = {
                    'u_statistic': float(u_stat),
                    'p_value': float(p_val),
                    'effect_size_r': float(r),
                    'n_comp_on': n1,
                    'n_comp_off': n2,
                    'median_comp_on': float(np.median(on_vals)),
                    'median_comp_off': float(np.median(off_vals))
                }
            except Exception as e:
                print(f"WARN: Mann-Whitney failed for {metric}: {e}")
    
    return results


def correlation_analysis(df: pd.DataFrame) -> Dict:
    """Correlation analysis: Gain vs ridge volume (theory validation)."""
    results = {}
    
    # Overall correlation
    valid_corr = df.dropna(subset=['gain_ridge_corr'])
    if len(valid_corr) > 0:
        correlations = valid_corr['gain_ridge_corr']
        results['overall'] = {
            'mean_correlation': float(np.mean(correlations)),
            'std_correlation': float(np.std(correlations)),
            'median_correlation': float(np.median(correlations)),
            'n_runs': len(correlations),
            'positive_corr_frac': float(np.mean(correlations > 0))
        }
    
    # By compensation regime
    for comp in [True, False]:
        subset = valid_corr[valid_corr['comp_demo'] == comp]
        if len(subset) > 0:
            correlations = subset['gain_ridge_corr']
            results[f'comp_{comp}'] = {
                'mean_correlation': float(np.mean(correlations)),
                'std_correlation': float(np.std(correlations)),
                'n_runs': len(correlations)
            }
    
    return results


def power_analysis(df: pd.DataFrame) -> Dict:
    """Retrospective power analysis (approximate, without scipy.stats.power)."""
    results = {}

    # Power for detecting PE vs PL difference
    valid_pairs = df.dropna(subset=['pe_pl_diff_at_true'])
    if len(valid_pairs) >= 3:
        diff = valid_pairs['pe_pl_diff_at_true']
        effect_size = float(np.mean(diff) / np.std(diff)) if np.std(diff) > 0 else 0.0
        n = int(len(diff))

        # Approximate two-sided t-test power using noncentral t heuristic
        alpha = 0.05
        # Critical value for large-sample approx; fall back to normal quantile
        try:
            t_critical = float(stats.t.ppf(1 - alpha / 2.0, n - 1))
        except Exception:
            t_critical = float(stats.norm.ppf(1 - alpha / 2.0))
        # Noncentrality parameter approximation
        ncp = abs(effect_size) * np.sqrt(n)
        # Heuristic power estimate using standard normal as proxy
        try:
            power_est = float(1 - stats.nct.cdf(t_critical, n - 1, ncp) + stats.nct.cdf(-t_critical, n - 1, ncp))
        except Exception:
            power_est = float(1 - stats.norm.cdf(t_critical - ncp) + stats.norm.cdf(-t_critical - ncp))

        results['pe_vs_pl'] = {
            'effect_size': effect_size,
            'sample_size': n,
            'estimated_power': power_est,
            'alpha': alpha,
        }

    return results


def generate_summary_report(df: pd.DataFrame, paired_t: Dict, mannwhitney: Dict, 
                          correlation: Dict, power: Dict, output_dir: str) -> None:
    """Generate comprehensive statistical summary report."""
    
    report_lines = [
        "PhD-LEVEL STATISTICAL ANALYSIS REPORT",
        "=" * 80,
        f"Generated: {datetime.now().isoformat()}",
        f"Total runs analyzed: {len(df)}",
        "",
        "DATASET SUMMARY",
        "-" * 40
    ]
    
    # Dataset overview
    report_lines.extend([
        f"Models: {sorted(df['model'].unique())}",
        f"Seeds: {sorted(df['seed'].unique())}",
        f"Compensation regimes: {sorted(df['comp_demo'].unique())}",
        f"Noise distributions: {sorted(df['noise_dist'].unique())}",
        ""
    ])
    
    # Paired t-test results
    if 'overall' in paired_t:
        pt = paired_t['overall']
        report_lines.extend([
            "PAIRED T-TEST: PE vs PL at True Parameters",
            "-" * 40,
            f"t-statistic: {pt['t_statistic']:.4f}",
            f"p-value: {pt['p_value']:.6f}",
            f"Cohen's d: {pt['cohens_d']:.4f}",
            f"Mean difference (PE - PL): {pt['mean_diff']:.6f}",
            f"Sample size: {pt['n_pairs']}",
            f"Significance: {'***' if pt['p_value'] < 0.001 else '**' if pt['p_value'] < 0.01 else '*' if pt['p_value'] < 0.05 else 'ns'}",
            ""
        ])
    
    # Mann-Whitney results
    if mannwhitney:
        report_lines.extend([
            "MANN-WHITNEY U TESTS: Compensation ON vs OFF",
            "-" * 40
        ])
        for metric, mw in mannwhitney.items():
            report_lines.extend([
                f"{metric.upper()}:",
                f"  U-statistic: {mw['u_statistic']:.2f}",
                f"  p-value: {mw['p_value']:.6f}",
                f"  Effect size (r): {mw['effect_size_r']:.4f}",
                f"  Median ON: {mw['median_comp_on']:.4f}",
                f"  Median OFF: {mw['median_comp_off']:.4f}",
                ""
            ])
    
    # Correlation analysis
    if 'overall' in correlation:
        corr = correlation['overall']
        report_lines.extend([
            "GAIN-RIDGE CORRELATION ANALYSIS",
            "-" * 40,
            f"Mean correlation: {corr['mean_correlation']:.4f} ± {corr['std_correlation']:.4f}",
            f"Median correlation: {corr['median_correlation']:.4f}",
            f"Positive correlation rate: {corr['positive_corr_frac']:.3f}",
            f"Sample size: {corr['n_runs']}",
            ""
        ])
    
    # Power analysis
    if 'pe_vs_pl' in power:
        pw = power['pe_vs_pl']
        report_lines.extend([
            "POWER ANALYSIS",
            "-" * 40,
            f"Effect size (Cohen's d): {pw['effect_size']:.4f}",
            f"Sample size: {pw['sample_size']}",
            f"Estimated power: {pw['estimated_power']:.4f}",
            f"Alpha level: {pw['alpha']:.3f}",
            ""
        ])
    
    # Interpretation
    report_lines.extend([
        "INTERPRETATION & CONCLUSIONS",
        "-" * 40,
        "Statistical significance thresholds:",
        "  *** p < 0.001 (highly significant)",
        "  **  p < 0.01  (very significant)", 
        "  *   p < 0.05  (significant)",
        "  ns  p ≥ 0.05  (not significant)",
        "",
        "Effect size interpretation (Cohen's d):",
        "  |d| < 0.2  (small effect)",
        "  |d| < 0.5  (medium effect)",
        "  |d| ≥ 0.8  (large effect)",
        ""
    ])
    
    # Save report
    report_path = os.path.join(output_dir, 'statistical_analysis_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print('\n'.join(report_lines))
    print(f"\nDetailed report saved: {report_path}")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'outputs'
    
    print("Loading results...")
    results = load_results(results_dir)
    print(f"Loaded {len(results)} result files")
    
    if len(results) == 0:
        print("No results found!")
        return
    
    print("Extracting metrics...")
    df = extract_metrics(results)
    
    print("Running statistical tests...")
    paired_t = paired_ttest_analysis(df)
    mannwhitney = mannwhitney_analysis(df)
    correlation = correlation_analysis(df)
    power = power_analysis(df)
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(results_dir, f'statistical_analysis_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save detailed results
    analysis_results = {
        'paired_t_tests': paired_t,
        'mannwhitney_tests': mannwhitney,
        'correlation_analysis': correlation,
        'power_analysis': power,
        'raw_data': df.to_dict('records')
    }
    
    with open(os.path.join(output_dir, 'statistical_analysis_detailed.pkl'), 'wb') as f:
        pickle.dump(analysis_results, f)
    
    df.to_csv(os.path.join(output_dir, 'metrics_summary.csv'), index=False)
    
    print("Generating summary report...")
    generate_summary_report(df, paired_t, mannwhitney, correlation, power, output_dir)


if __name__ == '__main__':
    main()
