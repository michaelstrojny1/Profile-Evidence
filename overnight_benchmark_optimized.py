#!/usr/bin/env python3
"""
OPTIMIZED OVERNIGHT BENCHMARK - CLEAN & COMPREHENSIVE
====================================================

This script runs the complete validation suite with all compensation detection
requirements clearly implemented:

1. Both 2D oscillator and SMIB models
2. Multiple seeds for robust statistics  
3. Compensation regimes: enabled/disabled
4. Noise distribution ablations: Gaussian, Laplace, Student-t
5. Full compensation detection artifacts per paper.tex requirements
6. Enhanced analysis and validation

Key compensation metrics validated:
- Delta curve: PE_norm - PL_norm (correction detection)
- Correction flags: Points where PE corrects PL underestimation
- Compensation gain: Laplace evidence - PL (theoretical validation)
- Ridge log-volume: Nuisance parameter uncertainty width
- Direct evidence: Compensation paths and clouds
- Fair comparison: Per-curve normalization, shared thresholds
"""

import os
import sys
import time
import numpy as np
import pickle
from datetime import datetime
from typing import List, Tuple, Dict, Any

def run_oscillator_config(seed: int, comp_demo: bool, noise_dist: str) -> Dict[str, Any]:
    """Run 2D oscillator with specific configuration."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] OSCILLATOR seed={seed}, comp={comp_demo}, noise={noise_dist}")
    
    try:
        import pe as bp
        
        # Configure for this run
        bp.CFG.enable_comp_demo = comp_demo
        bp.CFG.zeta_noise_dist = noise_dist
        # Strict control: disable correlation and anisotropy when compensation is OFF
        if not comp_demo:
            try:
                bp.CFG.enable_q_corr = False
                bp.CFG.r_std2 = bp.CFG.r_std
            except Exception:
                pass
        
        # Run analysis with plots and files enabled for complete artifacts
        # Prefix and route artifacts into timestamped output directory
        run_dir = os.environ.get('RUN_DIR', '.')
        os.environ['RUN_PREFIX'] = os.path.join(run_dir, f"oscillator_s{seed}_{comp_demo}_{noise_dist}_")
        results = bp.run_bulletproof(seed=seed, make_plots=True, write_files=True)
        results['model'] = 'oscillator'
        results['seed'] = seed
        results['config'] = {'enable_comp_demo': comp_demo, 'zeta_noise_dist': noise_dist}
        
        print(f"  SUCCESS: OSCILLATOR seed={seed} completed with all artifacts")
        return results
        
    except Exception as e:
        print(f"  ERROR: OSCILLATOR seed={seed} failed: {e}")
        return None

def run_smib_config(seed: int, comp_demo: bool, noise_dist: str) -> Dict[str, Any]:
    """Run SMIB with specific configuration."""
    print(f"[{datetime.now().strftime('%H:%M:%S')}] SMIB seed={seed}, comp={comp_demo}, noise={noise_dist}")
    
    try:
        # Load SMIB code dynamically (updated filename)
        with open('pe_smib.py', 'r') as f:
            smib_code = f.read()
        
        # Modify configuration
        modified_code = smib_code.replace(
            'enable_comp_demo: bool = True',
            f'enable_comp_demo: bool = {comp_demo}'
        ).replace(
            "zeta_noise_dist: str = 'gaussian'",
            f"zeta_noise_dist: str = '{noise_dist}'"
        )
        # Strict control: when compensation is OFF, ensure matched-model settings
        if not comp_demo:
            modified_code = modified_code.replace(
                'enable_q_corr: bool = True',
                'enable_q_corr: bool = False'
            ).replace(
                'r_std2: float = 0.40',
                'r_std2: float = 0.05'
            )
        
        # Execute in isolated namespace
        namespace = {}
        exec(modified_code, namespace)
        
        # Run analysis with plots and files enabled for complete artifacts
        run_dir = os.environ.get('RUN_DIR', '.')
        os.environ['RUN_PREFIX'] = os.path.join(run_dir, f"smib_s{seed}_{comp_demo}_{noise_dist}_")
        results = namespace['run_bulletproof'](seed=seed, make_plots=True, write_files=True)
        results['model'] = 'smib'
        results['seed'] = seed
        results['config'] = {'enable_comp_demo': comp_demo, 'zeta_noise_dist': noise_dist}
        
        print(f"  SUCCESS: SMIB seed={seed} completed with all artifacts")
        return results
        
    except Exception as e:
        print(f"  ERROR: SMIB seed={seed} failed: {e}")
        return None

def validate_compensation_artifacts(results: Dict[str, Any]) -> bool:
    """Validate that results contain all required compensation detection artifacts."""
    required_keys = [
        'delta_norm',        # PE_norm - PL_norm (key compensation indicator)
        'comp_gains',        # Laplace evidence - PL (theoretical validation)
        'ridge_logvols',     # Nuisance parameter ridge width
        'ess_rels',          # Sampling reliability
        'psi_mle_arr',       # MLE nuisance trajectories (for compensation_path.png)
        'psi_map_arr',       # MAP nuisance trajectories (for compensation_path.png)
        'all_is_samples',    # IS samples (for compensation_cloud plots)
        'all_is_weights'     # IS weights (for compensation_cloud plots)
    ]
    
    missing_keys = [key for key in required_keys if key not in results]
    
    if missing_keys:
        print(f"    WARNING: Missing compensation metrics: {missing_keys}")
        return False
    else:
        print(f"    VALIDATED: All compensation detection artifacts present")
        return True

def analyze_compensation_detection(all_results: List[Dict[str, Any]], output_dir: str):
    """Comprehensive compensation detection analysis."""
    print("\n" + "=" * 80)
    print("COMPENSATION DETECTION ANALYSIS")
    print("=" * 80)
    
    # Group results by configuration
    config_stats = {}
    for result in all_results:
        if result is None:
            continue
            
        model = result['model']
        config = result['config']
        comp_demo = config['enable_comp_demo']
        noise_dist = config['zeta_noise_dist']
        
        key = (model, comp_demo, noise_dist)
        if key not in config_stats:
            config_stats[key] = []
        config_stats[key].append(result)
    
    analysis_summary = []
    analysis_summary.append("COMPENSATION DETECTION VALIDATION REPORT")
    analysis_summary.append("=" * 80)
    analysis_summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    analysis_summary.append("")
    
    # Analyze each configuration
    for (model, comp_demo, noise_dist), results_list in config_stats.items():
        if not results_list:
            continue
            
        analysis_summary.append(f"Configuration: {model.upper()} | Compensation={comp_demo} | Noise={noise_dist}")
        analysis_summary.append("-" * 60)
        
        # Extract compensation metrics
        delta_fracs = []
        comp_gains_all = []
        correction_counts = []
        overestimation_counts = []
        ess_means = []
        
        for result in results_list:
            if 'delta_norm' in result:
                delta_norm = np.array(result['delta_norm'])
                delta_fracs.append(np.mean(delta_norm > 0.05))
                correction_counts.append(np.sum(delta_norm > 0.05))
                overestimation_counts.append(np.sum(delta_norm < -0.05))
            
            if 'comp_gains' in result:
                comp_gains_all.extend(result['comp_gains'])
            
            if 'ess_rels' in result:
                ess_means.append(np.mean(result['ess_rels']))
        
        # Configuration-level statistics
        if delta_fracs:
            mean_correction_frac = np.mean(delta_fracs)
            std_correction_frac = np.std(delta_fracs)
            mean_correction_count = np.mean(correction_counts)
            mean_overest_count = np.mean(overestimation_counts)
            
            analysis_summary.append(f"  PE Correction Rate: {mean_correction_frac:.3f} ± {std_correction_frac:.3f}")
            analysis_summary.append(f"  Mean correction points: {mean_correction_count:.1f}")
            analysis_summary.append(f"  Mean overestimation points: {mean_overest_count:.1f}")
        
        if comp_gains_all:
            comp_gains_arr = np.array(comp_gains_all)
            positive_gain_frac = np.mean(comp_gains_arr > 0)
            analysis_summary.append(f"  Positive compensation gain rate: {positive_gain_frac:.3f}")
            analysis_summary.append(f"  Mean compensation gain: {np.mean(comp_gains_arr):.3f}")
        
        if ess_means:
            analysis_summary.append(f"  Mean ESS: {np.mean(ess_means):.3f} ± {np.std(ess_means):.3f}")
        
        analysis_summary.append("")
    
    # Cross-model consistency
    analysis_summary.append("CROSS-MODEL COMPENSATION CONSISTENCY")
    analysis_summary.append("=" * 40)
    
    # Compare compensation enabled vs disabled
    comp_enabled = [r for r in all_results if r and r['config']['enable_comp_demo']]
    comp_disabled = [r for r in all_results if r and not r['config']['enable_comp_demo']]
    
    if comp_enabled and comp_disabled:
        enabled_fracs = [np.mean(np.array(r['delta_norm']) > 0.05) for r in comp_enabled if 'delta_norm' in r]
        disabled_fracs = [np.mean(np.array(r['delta_norm']) > 0.05) for r in comp_disabled if 'delta_norm' in r]
        
        if enabled_fracs and disabled_fracs:
            analysis_summary.append(f"Compensation ENABLED: {np.mean(enabled_fracs):.3f} ± {np.std(enabled_fracs):.3f}")
            analysis_summary.append(f"Compensation DISABLED: {np.mean(disabled_fracs):.3f} ± {np.std(disabled_fracs):.3f}")
            analysis_summary.append(f"Compensation effect size: {np.mean(enabled_fracs) - np.mean(disabled_fracs):+.3f}")
    
    analysis_summary.append("")
    
    # Key findings
    analysis_summary.append("KEY VALIDATION FINDINGS")
    analysis_summary.append("=" * 40)
    
    all_delta_fracs = []
    all_comp_gains = []
    all_ess_values = []
    
    for result in all_results:
        if result is None:
            continue
        if 'delta_norm' in result:
            delta_norm = np.array(result['delta_norm'])
            all_delta_fracs.append(np.mean(delta_norm > 0.05))
        if 'comp_gains' in result:
            all_comp_gains.extend(result['comp_gains'])
        if 'ess_rels' in result:
            all_ess_values.extend(result['ess_rels'])
    
    if all_delta_fracs:
        overall_correction_rate = np.mean(all_delta_fracs)
        analysis_summary.append(f"Overall PE correction rate: {overall_correction_rate:.3f}")
        
        if overall_correction_rate > 0.15:
            analysis_summary.append("SIGNIFICANT: PE corrects PL at >15% of parameter points")
        else:
            analysis_summary.append("! WEAK: PE corrects PL at <15% of parameter points")
    
    if all_comp_gains:
        positive_gain_rate = np.mean(np.array(all_comp_gains) > 0)
        analysis_summary.append(f"Positive compensation gain rate: {positive_gain_rate:.3f}")
        
        if positive_gain_rate > 0.6:
            analysis_summary.append("VALIDATION: >60% positive compensation gains")
        else:
            analysis_summary.append("! THEORY CONCERN: <60% positive compensation gains")
    
    if all_ess_values:
        reliable_rate = np.mean(np.array(all_ess_values) > 0.1)
        analysis_summary.append(f"Reliable sampling rate (ESS>0.1): {reliable_rate:.3f}")
        
        if reliable_rate > 0.9:
            analysis_summary.append("RELIABLE: >90% of samples have adequate ESS")
        else:
            analysis_summary.append("! UNRELIABLE: <90% of samples have adequate ESS")
    
    analysis_summary.append("")
    analysis_summary.append("INTERPRETATION:")
    analysis_summary.append("- PE correction rate measures how often PE fixes PL's underestimation")
    analysis_summary.append("- Positive compensation gains validate the theoretical mechanism")
    analysis_summary.append("- High ESS ensures corrections are statistically reliable")
    analysis_summary.append("- Cross-model consistency demonstrates generalizability")
    
    # Save analysis
    analysis_file = os.path.join(output_dir, "compensation_validation_report.txt")
    with open(analysis_file, 'w', encoding='utf-8') as f:
        f.write("\n".join(analysis_summary))
    
    print(f"Saved compensation validation report: {analysis_file}")
    for line in analysis_summary:
        print(line)

def main():
    """Run optimized overnight benchmark with full compensation validation."""
    print("=" * 80)
    print("OPTIMIZED OVERNIGHT BENCHMARK - COMPENSATION DETECTION VALIDATION")
    print("=" * 80)
    
    # Create timestamped output directory under RUN_DIR_BASE (default: outputs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir_base = os.environ.get('RUN_DIR_BASE', 'outputs')
    os.makedirs(run_dir_base, exist_ok=True)
    output_dir = os.path.join(run_dir_base, f"overnight_benchmark_optimized_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    os.environ['RUN_DIR'] = output_dir
    
    # Optimized configuration matrix (focused but comprehensive)
    seeds = list(range(6))  # 6 seeds for good statistics without excessive runtime
    comp_demos = [True, False]  # Compensation enabled/disabled
    noise_dists = ['gaussian', 'laplace', 'student_t']  # All noise distributions
    models = ['oscillator', 'smib']  # Both physical systems
    
    total_configs = len(models) * len(seeds) * len(comp_demos) * len(noise_dists)
    print(f"Total configurations: {total_configs}")
    print(f"Output directory: {output_dir}")
    print(f"Models: {models}")
    print(f"Seeds: {seeds}")
    print(f"Compensation: {comp_demos}")
    print(f"Noise distributions: {noise_dists}")
    print()
    print("COMPENSATION DETECTION REQUIREMENTS:")
    print("- Delta curve: PE_norm - PL_norm")
    print("- Correction flags: Points where PE > PL")
    print("- Direct evidence: Compensation paths and clouds")
    print("- Fair comparison: Per-curve normalization")
    print("- Validation: Compensation gain analysis")
    print()
    
    all_results = []
    completed = 0
    failed = 0
    start_time = time.time()
    
    # Run all configurations
    for model in models:
        for seed in seeds:
            for comp_demo in comp_demos:
                for noise_dist in noise_dists:
                    
                    print(f"\nConfiguration {completed+1}/{total_configs}:")
                    
                    # Run appropriate model
                    if model == 'oscillator':
                        result = run_oscillator_config(seed, comp_demo, noise_dist)
                    else:  # smib
                        result = run_smib_config(seed, comp_demo, noise_dist)
                    
                    # Validate and store result
                    if result is not None:
                        # Validate compensation artifacts
                        if validate_compensation_artifacts(result):
                            all_results.append(result)
                            
                            # Save individual result
                            result_file = os.path.join(
                                output_dir, 
                                f"{model}_s{seed}_{comp_demo}_{noise_dist}.pkl"
                            )
                            with open(result_file, 'wb') as f:
                                pickle.dump(result, f)
                        else:
                            print(f"    WARNING: Result missing compensation artifacts")
                            failed += 1
                    else:
                        failed += 1
                    
                    completed += 1
                    elapsed = time.time() - start_time
                    eta = elapsed * (total_configs - completed) / completed if completed > 0 else 0
                    
                    print(f"Progress: {completed}/{total_configs} ({100*completed/total_configs:.1f}%) | "
                          f"Success: {len(all_results)} | Failed: {failed} | "
                          f"Elapsed: {elapsed/3600:.1f}h | ETA: {eta/3600:.1f}h")
                    print("-" * 60)
    
    # Save consolidated results
    summary_file = os.path.join(output_dir, "optimized_benchmark_results.pkl")
    with open(summary_file, 'wb') as f:
        pickle.dump(all_results, f)
    
    # Comprehensive compensation analysis
    if all_results:
        analyze_compensation_detection(all_results, output_dir)
    
    print(f"\nOPTIMIZED BENCHMARK COMPLETE!")
    print(f"Total runtime: {elapsed/3600:.2f} hours")
    print(f"Results saved in: {output_dir}")
    print(f"Successful configurations: {len(all_results)}/{total_configs}")
    print(f"All compensation detection artifacts validated and saved!")

if __name__ == '__main__':
    main()
