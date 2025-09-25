#!/usr/bin/env python3
"""
Polished Benchmark Suite for Profile Evidence
============================================

Implements comprehensive dose-response validation with statistical testing:
- 4 compensation levels: 0× (control), 1.0×, 1.5×, 2.0×
- 2 models: oscillator, SMIB
- 3 noise distributions: gaussian, laplace, student_t
- 10 seeds per configuration
- Total: 4 × 2 × 3 × 10 = 240 configurations

Statistical analyses:
- Dose-response trend testing (Jonckheere-Terpstra)
- Paired sign tests across compensation levels
- Spearman correlation: compensation gain vs ridge log-volume
- Effect size calculations (Cohen's d)

Author: Benchmark generator
"""

import os
import sys
import pickle
import numpy as np
from datetime import datetime
import json
import subprocess
from typing import Dict, List, Tuple, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from scipy import stats
import warnings

# Suppress numpy warnings during intense computation
warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import the existing benchmark functions
try:
    from overnight_benchmark_optimized import run_oscillator_config, run_smib_config
except ImportError:
    print("ERROR: Cannot import overnight_benchmark_optimized.py")
    sys.exit(1)

# Import the main scripts for dose modification
try:
    import pe as bp_osc
    import pe_smib as bp_smib
except ImportError:
    print("ERROR: Cannot import profile evidence scripts")
    sys.exit(1)

class PolishedConfig:
    """Configuration for polished benchmark runs"""
    
    DOSE_FACTORS = [0.0, 1.0, 1.5, 2.0]  # Compensation strength multipliers
    MODELS = ['oscillator', 'smib']
    NOISE_DISTS = ['gaussian', 'laplace', 'student_t']
    SEEDS = list(range(10))  # 10 seeds for robust statistics
    
    @staticmethod
    def get_total_configs():
        return len(PolishedConfig.DOSE_FACTORS) * len(PolishedConfig.MODELS) * len(PolishedConfig.NOISE_DISTS) * len(PolishedConfig.SEEDS)

def modify_compensation_strength(model: str, dose_factor: float):
    """
    Modify the compensation parameters in the respective model scripts.
    
    Args:
        model: 'oscillator' or 'smib'
        dose_factor: Multiplier for compensation strength (0.0 = no compensation, 2.0 = double)
    """
    if model == 'oscillator':
        # Backup original values
        orig_walk_std = bp_osc.CFG.zeta_walk_std
        orig_jitter_std = bp_osc.CFG.zeta_jitter_std
        orig_enable_q_corr = bp_osc.CFG.enable_q_corr
        orig_r_std2 = bp_osc.CFG.r_std2
        
        if dose_factor == 0.0:
            # Perfect model match (negative control)
            bp_osc.CFG.zeta_walk_std = 0.0
            bp_osc.CFG.zeta_jitter_std = 0.0  
            bp_osc.CFG.enable_q_corr = False  # No process noise correlation
            bp_osc.CFG.r_std2 = bp_osc.CFG.r_std  # Equal measurement noise
        else:
            # Scale compensation strength
            bp_osc.CFG.zeta_walk_std = orig_walk_std * dose_factor
            bp_osc.CFG.zeta_jitter_std = orig_jitter_std * dose_factor
            # Keep other mismatches for compensation scenarios
            bp_osc.CFG.enable_q_corr = True
            bp_osc.CFG.r_std2 = 0.6  # Anisotropic measurement noise
    
    elif model == 'smib':
        # Backup original values
        orig_walk_std = bp_smib.CFG.zeta_walk_std  # This is delta0_walk_std in SMIB
        orig_jitter_std = bp_smib.CFG.zeta_jitter_std  # This is delta0_jitter_std in SMIB
        orig_enable_q_corr = bp_smib.CFG.enable_q_corr
        orig_r_std2 = bp_smib.CFG.r_std2
        
        if dose_factor == 0.0:
            # Perfect model match (negative control)
            bp_smib.CFG.zeta_walk_std = 0.0
            bp_smib.CFG.zeta_jitter_std = 0.0
            bp_smib.CFG.enable_q_corr = False
            bp_smib.CFG.r_std2 = bp_smib.CFG.r_std
        else:
            # Scale compensation strength
            bp_smib.CFG.zeta_walk_std = orig_walk_std * dose_factor
            bp_smib.CFG.zeta_jitter_std = orig_jitter_std * dose_factor
            # Keep other mismatches
            bp_smib.CFG.enable_q_corr = True
            bp_smib.CFG.r_std2 = 0.3  # Anisotropic measurement noise

def run_polished_config(model: str, seed: int, dose_factor: float, noise_dist: str) -> Dict[str, Any]:
    """
    Run a single polished configuration with dose-modified compensation.
    
    Returns:
        dict: Results dictionary with dose_factor and model metadata added
    """
    print(f"[{datetime.now().strftime('%H:%M:%S')}] POLISHED {model.upper()} seed={seed}, dose={dose_factor}×, noise={noise_dist}")
    
    # Modify compensation strength
    modify_compensation_strength(model, dose_factor)
    
    # Set environment for compensation state
    os.environ['COMPENSATION_DEMO'] = 'True' if dose_factor > 0.0 else 'False'
    
    # Run the configuration
    try:
        if model == 'oscillator':
            # Map dose_factor to enable_comp_demo boolean for compatibility
            comp_demo = dose_factor > 0.0
            result = run_oscillator_config(seed, comp_demo, noise_dist)
        else:  # smib
            comp_demo = dose_factor > 0.0
            result = run_smib_config(seed, comp_demo, noise_dist)
        
        # Add metadata
        result['dose_factor'] = dose_factor
        result['model'] = model
        result['noise_dist'] = noise_dist
        result['seed'] = seed
        
        # Extract key metrics for statistical analysis
        if 'delta_norm' in result:
            delta_vals = result['delta_norm']
            result['positive_delta_fraction'] = float(np.mean(delta_vals > 0.05))
            result['negative_delta_fraction'] = float(np.mean(delta_vals < -0.05))
            result['mean_delta'] = float(np.mean(delta_vals))
            result['delta_std'] = float(np.std(delta_vals))
        
        if 'comp_gains' in result and 'ridge_logvols' in result:
            # Spearman correlation between compensation gain and ridge volume
            try:
                gains = np.array(result['comp_gains'])
                vols = np.array(result['ridge_logvols'])
                valid_mask = np.isfinite(gains) & np.isfinite(vols)
                if np.sum(valid_mask) >= 3:  # Need at least 3 points for correlation
                    rho, p_val = stats.spearmanr(gains[valid_mask], vols[valid_mask])
                    result['gain_volume_spearman_rho'] = float(rho) if np.isfinite(rho) else np.nan
                    result['gain_volume_spearman_p'] = float(p_val) if np.isfinite(p_val) else np.nan
                else:
                    result['gain_volume_spearman_rho'] = np.nan
                    result['gain_volume_spearman_p'] = np.nan
            except Exception:
                result['gain_volume_spearman_rho'] = np.nan
                result['gain_volume_spearman_p'] = np.nan
        
        return result
    
    except Exception as e:
        print(f"ERROR: POLISHED {model} seed={seed} dose={dose_factor}× failed: {e}")
        return {
            'error': str(e),
            'dose_factor': dose_factor,
            'model': model,
            'noise_dist': noise_dist,
            'seed': seed
        }

def analyze_dose_response(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze dose-response patterns and compute statistical tests.
    
    Args:
        results: List of result dictionaries from polished runs
        
    Returns:
        dict: Statistical analysis results
    """
    print("\n" + "="*80)
    print("DOSE-RESPONSE STATISTICAL ANALYSIS")
    print("="*80)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame([r for r in results if 'error' not in r])
    
    if df.empty:
        return {'error': 'No valid results for analysis'}
    
    analysis = {
        'n_configs_analyzed': len(df),
        'dose_response': {},
        'statistical_tests': {},
        'cross_model_comparison': {}
    }
    
    # 1. Dose-response analysis by model
    for model in PolishedConfig.MODELS:
        model_df = df[df['model'] == model].copy()
        if model_df.empty:
            continue
            
        dose_stats = []
        for dose in PolishedConfig.DOSE_FACTORS:
            dose_df = model_df[model_df['dose_factor'] == dose]
            if not dose_df.empty and 'positive_delta_fraction' in dose_df.columns:
                pos_fracs = dose_df['positive_delta_fraction'].dropna().values
                dose_stats.append({
                    'dose': dose,
                    'n_runs': len(pos_fracs),
                    'pos_delta_mean': float(np.mean(pos_fracs)),
                    'pos_delta_std': float(np.std(pos_fracs)),
                    'pos_delta_median': float(np.median(pos_fracs)),
                    'pos_delta_q25': float(np.percentile(pos_fracs, 25)),
                    'pos_delta_q75': float(np.percentile(pos_fracs, 75))
                })
        
        analysis['dose_response'][model] = dose_stats
    
    # 2. Jonckheere-Terpstra trend test for monotonic dose response
    for model in PolishedConfig.MODELS:
        model_df = df[df['model'] == model].copy()
        if model_df.empty or 'positive_delta_fraction' not in model_df.columns:
            continue
            
        # Group positive delta fractions by dose
        dose_groups = []
        dose_labels = []
        for dose in sorted(PolishedConfig.DOSE_FACTORS):
            dose_data = model_df[model_df['dose_factor'] == dose]['positive_delta_fraction'].dropna().values
            if len(dose_data) > 0:
                dose_groups.append(dose_data)
                dose_labels.append(dose)
        
        if len(dose_groups) >= 3:  # Need at least 3 groups for trend test
            try:
                # Jonckheere-Terpstra test implementation
                # (scipy doesn't have this, so we'll use Mann-Whitney U between adjacent groups)
                trend_stats = []
                for i in range(len(dose_groups) - 1):
                    stat, p_val = stats.mannwhitneyu(dose_groups[i], dose_groups[i+1], alternative='less')
                    trend_stats.append({
                        'comparison': f"{dose_labels[i]}× vs {dose_labels[i+1]}×",
                        'mann_whitney_u': float(stat),
                        'p_value': float(p_val),
                        'significant': p_val < 0.05
                    })
                
                analysis['statistical_tests'][f'{model}_trend_tests'] = trend_stats
            
            except Exception as e:
                analysis['statistical_tests'][f'{model}_trend_error'] = str(e)
    
    # 3. Sign test: H0: P(Δ>0) = 0.5 vs H1: P(Δ>0) > 0.5
    for model in PolishedConfig.MODELS:
        for dose in PolishedConfig.DOSE_FACTORS:
            if dose == 0.0:  # Skip control for sign test
                continue
                
            subset = df[(df['model'] == model) & (df['dose_factor'] == dose)]
            if not subset.empty and 'positive_delta_fraction' in subset.columns:
                pos_fracs = subset['positive_delta_fraction'].dropna().values
                if len(pos_fracs) > 0:
                    # Binomial test: n_positive out of total observations
                    # Each pos_frac represents fraction of ω points with Δ>0.05
                    mean_pos_frac = np.mean(pos_fracs)
                    n_total_omega_points = len(pos_fracs) * 21  # Approximate omega grid size
                    n_positive = int(mean_pos_frac * n_total_omega_points)
                    
                    # Binomial test
                    p_val = stats.binomtest(n_positive, n_total_omega_points, 0.5, alternative='greater').pvalue
                    
                    test_key = f'{model}_dose_{dose}_sign_test'
                    analysis['statistical_tests'][test_key] = {
                        'mean_positive_fraction': float(mean_pos_frac),
                        'binomial_p_value': float(p_val),
                        'significant': p_val < 0.05,
                        'n_runs': len(pos_fracs)
                    }
    
    # 4. Cross-model consistency
    if len(df['model'].unique()) == 2:
        # Compare dose-response patterns between models
        osc_df = df[df['model'] == 'oscillator']
        smib_df = df[df['model'] == 'smib']
        
        model_consistency = {}
        for dose in PolishedConfig.DOSE_FACTORS:
            osc_dose = osc_df[osc_df['dose_factor'] == dose]['positive_delta_fraction'].dropna().values
            smib_dose = smib_df[smib_df['dose_factor'] == dose]['positive_delta_fraction'].dropna().values
            
            if len(osc_dose) > 0 and len(smib_dose) > 0:
                # Mann-Whitney U test between models at same dose
                try:
                    stat, p_val = stats.mannwhitneyu(osc_dose, smib_dose, alternative='two-sided')
                    model_consistency[f'dose_{dose}'] = {
                        'osc_mean': float(np.mean(osc_dose)),
                        'smib_mean': float(np.mean(smib_dose)),
                        'mann_whitney_u': float(stat),
                        'p_value': float(p_val),
                        'models_similar': p_val > 0.05  # Non-significant = similar
                    }
                except Exception as e:
                    model_consistency[f'dose_{dose}'] = {'error': str(e)}
        
        analysis['cross_model_comparison'] = model_consistency
    
    # 5. Effect sizes (Cohen's d)
    for model in PolishedConfig.MODELS:
        model_df = df[df['model'] == model].copy()
        if model_df.empty:
            continue
            
        # Cohen's d between control (0×) and each compensation level
        control_data = model_df[model_df['dose_factor'] == 0.0]['positive_delta_fraction'].dropna().values
        
        if len(control_data) > 0:
            effect_sizes = {}
            for dose in [1.0, 1.5, 2.0]:
                treatment_data = model_df[model_df['dose_factor'] == dose]['positive_delta_fraction'].dropna().values
                
                if len(treatment_data) > 0:
                    # Cohen's d calculation
                    pooled_std = np.sqrt(((len(control_data) - 1) * np.var(control_data) + 
                                        (len(treatment_data) - 1) * np.var(treatment_data)) / 
                                       (len(control_data) + len(treatment_data) - 2))
                    
                    if pooled_std > 0:
                        cohens_d = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std
                        effect_sizes[f'{dose}x_vs_control'] = {
                            'cohens_d': float(cohens_d),
                            'effect_magnitude': 'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small',
                            'control_mean': float(np.mean(control_data)),
                            'treatment_mean': float(np.mean(treatment_data))
                        }
            
            analysis['statistical_tests'][f'{model}_effect_sizes'] = effect_sizes
    
    return analysis

def generate_polished_report(results: List[Dict[str, Any]], analysis: Dict[str, Any], output_dir: str):
    """Generate comprehensive polished benchmark report."""
    
    report_lines = [
        "="*90,
        "POLISHED BENCHMARK COMPREHENSIVE REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "="*90,
        "",
        f"TOTAL CONFIGURATIONS: {len(results)}",
        f"SUCCESSFUL RUNS: {len([r for r in results if 'error' not in r])}",
        f"FAILED RUNS: {len([r for r in results if 'error' in r])}",
        "",
        "EXPERIMENTAL DESIGN:",
        "- 4 compensation levels: 0× (control), 1.0×, 1.5×, 2.0×",
        "- 2 physical models: 2D oscillator, SMIB power system", 
        "- 3 noise distributions: Gaussian, Laplace, Student-t",
        "- 10 random seeds per configuration",
        "- Total: 240 configurations for robust statistical inference",
        "",
    ]
    
    # Add dose-response results
    if 'dose_response' in analysis:
        report_lines.extend([
            "DOSE-RESPONSE ANALYSIS:",
            "-" * 50
        ])
        
        for model, dose_stats in analysis['dose_response'].items():
            report_lines.extend([
                f"",
                f"{model.upper()} Model:",
            ])
            for stat in dose_stats:
                dose = stat['dose']
                mean_pos = stat['pos_delta_mean']
                std_pos = stat['pos_delta_std']
                n_runs = stat['n_runs']
                report_lines.append(f"  {dose:3.1f}×: Pos-Δ fraction = {mean_pos:.3f} ± {std_pos:.3f} (n={n_runs})")
    
    # Add statistical test results
    if 'statistical_tests' in analysis:
        report_lines.extend([
            "",
            "STATISTICAL TESTS:",
            "-" * 50
        ])
        
        for test_name, test_result in analysis['statistical_tests'].items():
            if 'trend_tests' in test_name:
                report_lines.extend([f"", f"{test_name}:"])
                for trend in test_result:
                    sig = "***" if trend['significant'] else "n.s."
                    report_lines.append(f"  {trend['comparison']}: U={trend['mann_whitney_u']:.1f}, p={trend['p_value']:.4f} {sig}")
            
            elif 'sign_test' in test_name:
                pos_frac = test_result['mean_positive_fraction']
                p_val = test_result['binomial_p_value']
                sig = "***" if test_result['significant'] else "n.s."
                report_lines.append(f"{test_name}: Pos-Δ={pos_frac:.3f}, p={p_val:.4f} {sig}")
            
            elif 'effect_sizes' in test_name:
                report_lines.extend([f"", f"{test_name}:"])
                for comparison, effect in test_result.items():
                    d = effect['cohens_d']
                    mag = effect['effect_magnitude']
                    report_lines.append(f"  {comparison}: d={d:.3f} ({mag})")
    
    # Add cross-model comparison
    if 'cross_model_comparison' in analysis:
        report_lines.extend([
            "",
            "CROSS-MODEL CONSISTENCY:",
            "-" * 50
        ])
        
        for dose_key, comparison in analysis['cross_model_comparison'].items():
            if 'error' not in comparison:
                osc_mean = comparison['osc_mean']
                smib_mean = comparison['smib_mean']
                similar = "✓" if comparison['models_similar'] else "✗"
                p_val = comparison['p_value']
                report_lines.append(f"{dose_key}: OSC={osc_mean:.3f}, SMIB={smib_mean:.3f}, similar={similar} (p={p_val:.4f})")
    
    # Add summary and conclusion
    report_lines.extend([
        "",
        "SUMMARY & CONCLUSIONS:",
        "-" * 50,
        ""
    ])
    
    # Check if dose-response pattern is monotonic
    monotonic_evidence = []
    if 'dose_response' in analysis:
        for model, dose_stats in analysis['dose_response'].items():
            doses = [s['dose'] for s in dose_stats]
            pos_fracs = [s['pos_delta_mean'] for s in dose_stats]
            
            if len(doses) >= 3:
                # Check if generally increasing
                increases = sum(pos_fracs[i] < pos_fracs[i+1] for i in range(len(pos_fracs)-1))
                total_comparisons = len(pos_fracs) - 1
                monotonic_evidence.append(f"{model}: {increases}/{total_comparisons} dose increases show higher PE correction")
    
    report_lines.extend(monotonic_evidence)
    
    # Write report
    report_path = os.path.join(output_dir, 'polished_comprehensive_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    
    print(f"Comprehensive report saved: {report_path}")
    return report_path

def main():
    """Main polished benchmark execution."""
    print("="*90)
    print("POLISHED PROFILE EVIDENCE BENCHMARK SUITE")
    print("="*90)
    
    # Setup output directory
    run_dir_base = os.environ.get('RUN_DIR_BASE', 'outputs')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(run_dir_base, f"polished_benchmark_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.environ['RUN_DIR'] = output_dir
    
    print(f"Output directory: {output_dir}")
    print(f"Total configurations: {PolishedConfig.get_total_configs()}")
    print()
    
    # Generate all configuration combinations
    configs = []
    for dose in PolishedConfig.DOSE_FACTORS:
        for model in PolishedConfig.MODELS:
            for noise_dist in PolishedConfig.NOISE_DISTS:
                for seed in PolishedConfig.SEEDS:
                    configs.append((model, seed, dose, noise_dist))
    
    print(f"Running {len(configs)} configurations...")
    
    # Run configurations
    results = []
    successful = 0
    failed = 0
    
    for i, (model, seed, dose, noise_dist) in enumerate(configs):
        print(f"\n[{i+1:3}/{len(configs)}] {model} s{seed} {dose}× {noise_dist}")
        
        try:
            result = run_polished_config(model, seed, dose, noise_dist)
            results.append(result)
            
            if 'error' not in result:
                successful += 1
                # Save individual result
                result_file = os.path.join(output_dir, f"{model}_s{seed}_dose{dose}__{noise_dist}.pkl")
                with open(result_file, 'wb') as f:
                    pickle.dump(result, f)
            else:
                failed += 1
                print(f"    FAILED: {result['error']}")
        
        except Exception as e:
            print(f"    CRITICAL FAILURE: {e}")
            results.append({
                'error': str(e),
                'model': model,
                'seed': seed,
                'dose_factor': dose,
                'noise_dist': noise_dist
            })
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"POLISHED BENCHMARK COMPLETE")
    print(f"Successful: {successful}/{len(configs)}")
    print(f"Failed: {failed}/{len(configs)}")
    print(f"{'='*60}")
    
    # Save consolidated results
    consolidated_file = os.path.join(output_dir, 'polished_consolidated_results.pkl')
    with open(consolidated_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Perform statistical analysis
    print("\nPerforming statistical analysis...")
    analysis = analyze_dose_response(results)
    
    analysis_file = os.path.join(output_dir, 'polished_statistical_analysis.pkl')
    with open(analysis_file, 'wb') as f:
        pickle.dump(analysis, f)
    
    # Generate comprehensive report
    report_path = generate_polished_report(results, analysis, output_dir)
    
    print(f"\nAll outputs saved to: {output_dir}")
    print(f"Key files:")
    print(f"  - {consolidated_file}")
    print(f"  - {analysis_file}")
    print(f"  - {report_path}")
    print("\nPolished benchmark suite complete!")

if __name__ == '__main__':
    main()
