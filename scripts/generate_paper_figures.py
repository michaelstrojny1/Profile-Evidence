#!/usr/bin/env python3
"""
Generate Publication-Quality Figures for PE vs PL Paper

Creates publication-ready figures with proper formatting, captions,
and statistical annotations for journal submission.

Usage:
  python -u scripts/generate_paper_figures.py [RESULTS_DIR] [OUTPUT_DIR]
"""

import os
import sys
import glob
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json


# Publication-quality plotting settings
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'dejavuserif',
    'axes.linewidth': 1.0,
    'axes.spines.left': True,
    'axes.spines.bottom': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'xtick.major.size': 4,
    'xtick.minor.size': 2,
    'ytick.major.size': 4,
    'ytick.minor.size': 2,
    'legend.frameon': True,
    'legend.fancybox': False,
    'legend.shadow': False,
    'legend.numpoints': 1,
    'legend.scatterpoints': 1,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
})


def load_statistical_analysis(results_dir: str) -> Optional[Dict]:
    """Load statistical analysis results if available."""
    analysis_files = glob.glob(os.path.join(results_dir, '**/statistical_analysis_detailed.pkl'), recursive=True)
    if analysis_files:
        latest_file = max(analysis_files, key=os.path.getmtime)
        with open(latest_file, 'rb') as f:
            return pickle.load(f)
    return None


def load_results_data(results_dir: str) -> pd.DataFrame:
    """Load and consolidate all PE results into DataFrame."""
    pkl_files = glob.glob(os.path.join(results_dir, '**', '*_pe_results.pkl'), recursive=True)
    
    rows = []
    for pkl_path in pkl_files:
        try:
            with open(pkl_path, 'rb') as f:
                res = pickle.load(f)
            
            cfg = res.get('config', {})
            
            # Extract key metrics
            row = {
                'model': res.get('model', 'unknown'),
                'seed': res.get('seed', -1),
                'comp_demo': cfg.get('enable_comp_demo', None),
                'noise_dist': cfg.get('zeta_noise_dist', 'unknown'),
                'omega_true': res.get('omega_true', np.nan),
                'omegas': res.get('omegas', []),
                'pl_vals': res.get('pl_vals', []),
                'pe_vals': res.get('pe_vals', []),
                'delta_norm': res.get('delta_norm', []),
                'ess_rels': res.get('ess_rels', []),
                'comp_gains': res.get('comp_gains', []),
                'ridge_logvols': res.get('ridge_logvols', []),
                'pl_ci': res.get('pl_ci', (np.nan, np.nan)),
                'pe_ci': res.get('pe_ci', (np.nan, np.nan)),
                'pkl_path': pkl_path
            }
            
            rows.append(row)
            
        except Exception as e:
            print(f"WARN: Failed to load {pkl_path}: {e}")
    
    return pd.DataFrame(rows)


def create_figure_1_multi_model_comparison(df: pd.DataFrame, output_dir: str) -> None:
    """Figure 1: Multi-model PE vs PL comparison with statistical significance."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    models = df['model'].unique()
    colors = {'oscillator': '#1f77b4', 'smib': '#ff7f0e'}
    
    for i, model in enumerate(models):
        if i >= 2:
            break
            
        model_data = df[df['model'] == model]
        
        # Compensation ON vs OFF comparison
        comp_on = model_data[model_data['comp_demo'] == True]
        comp_off = model_data[model_data['comp_demo'] == False]
        
        ax1 = axes[i, 0]
        ax2 = axes[i, 1]
        
        # Plot individual curves (compensation ON)
        if len(comp_on) > 0:
            for idx, row in comp_on.iterrows():
                omegas = np.array(row['omegas'])
                pl_norm = np.array(row['pl_vals']) - np.max(row['pl_vals'])
                pe_norm = np.array(row['pe_vals']) - np.max(row['pe_vals'])
                
                if len(omegas) > 0:
                    ax1.plot(omegas, pl_norm, 'b-', alpha=0.3, linewidth=1)
                    ax1.plot(omegas, pe_norm, 'r-', alpha=0.3, linewidth=1)
            
            # Mean curves
            all_omegas = []
            all_pl_norm = []
            all_pe_norm = []
            
            for idx, row in comp_on.iterrows():
                if len(row['omegas']) > 0:
                    all_omegas.append(np.array(row['omegas']))
                    all_pl_norm.append(np.array(row['pl_vals']) - np.max(row['pl_vals']))
                    all_pe_norm.append(np.array(row['pe_vals']) - np.max(row['pe_vals']))
            
            if all_omegas:
                # Interpolate to common grid for averaging
                common_omega = np.linspace(
                    max(omega.min() for omega in all_omegas),
                    min(omega.max() for omega in all_omegas),
                    25
                )
                
                pl_interp = []
                pe_interp = []
                
                for omega_arr, pl_arr, pe_arr in zip(all_omegas, all_pl_norm, all_pe_norm):
                    pl_interp.append(np.interp(common_omega, omega_arr, pl_arr))
                    pe_interp.append(np.interp(common_omega, omega_arr, pe_arr))
                
                mean_pl = np.mean(pl_interp, axis=0)
                mean_pe = np.mean(pe_interp, axis=0)
                std_pl = np.std(pl_interp, axis=0)
                std_pe = np.std(pe_interp, axis=0)
                
                ax1.plot(common_omega, mean_pl, 'b-', linewidth=3, label='PL (mean)')
                ax1.plot(common_omega, mean_pe, 'r-', linewidth=3, label='PE (mean)')
                ax1.fill_between(common_omega, mean_pl - std_pl, mean_pl + std_pl, 
                               alpha=0.2, color='blue')
                ax1.fill_between(common_omega, mean_pe - std_pe, mean_pe + std_pe, 
                               alpha=0.2, color='red')
        
        ax1.axhline(-0.5 * 3.841458820694124, color='gray', linestyle=':', alpha=0.7, 
                   label='95% threshold')
        ax1.set_xlabel('Parameter of Interest (ω)')
        ax1.set_ylabel('Normalized Log-Likelihood/Evidence')
        ax1.set_title(f'{model.upper()}: Compensation ON')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Delta curves (PE - PL)
        if len(comp_on) > 0:
            for idx, row in comp_on.iterrows():
                delta = np.array(row['delta_norm'])
                omegas = np.array(row['omegas'])
                
                if len(delta) > 0 and len(omegas) > 0:
                    ax2.plot(omegas, delta, alpha=0.5, linewidth=1.5, color=colors.get(model, 'gray'))
            
            # Mean delta curve
            if all_omegas:
                delta_interp = []
                for idx, row in comp_on.iterrows():
                    delta = np.array(row['delta_norm'])
                    omega_arr = np.array(row['omegas'])
                    if len(delta) > 0 and len(omega_arr) > 0:
                        delta_interp.append(np.interp(common_omega, omega_arr, delta))
                
                if delta_interp:
                    mean_delta = np.mean(delta_interp, axis=0)
                    std_delta = np.std(delta_interp, axis=0)
                    
                    ax2.plot(common_omega, mean_delta, color=colors.get(model, 'gray'), 
                            linewidth=3, label=f'{model.upper()} (mean)')
                    ax2.fill_between(common_omega, mean_delta - std_delta, mean_delta + std_delta,
                                   alpha=0.3, color=colors.get(model, 'gray'))
        
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5, linewidth=0.8)
        ax2.axhline(0.05, color='green', linestyle='--', alpha=0.7, label='Correction threshold')
        ax2.axhline(-0.05, color='orange', linestyle='--', alpha=0.7, label='Overestimation threshold')
        ax2.set_xlabel('Parameter of Interest (ω)')
        ax2.set_ylabel('Δ = PE_norm - PL_norm')
        ax2.set_title(f'{model.upper()}: Compensation Detection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle('Figure 1: Multi-Model PE vs PL Comparison', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_1_multi_model_comparison.png'))
    plt.savefig(os.path.join(output_dir, 'figure_1_multi_model_comparison.pdf'))
    plt.close()
    
    print("Generated Figure 1: Multi-model comparison")


def create_figure_2_statistical_significance(df: pd.DataFrame, stats_analysis: Optional[Dict], output_dir: str) -> None:
    """Figure 2: Statistical significance heatmap and effect sizes."""
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Compensation detection rates by model and noise
    comp_rates = []
    models = sorted(df['model'].unique())
    noise_types = sorted(df['noise_dist'].unique())
    
    rate_matrix = np.zeros((len(models), len(noise_types)))
    
    for i, model in enumerate(models):
        for j, noise in enumerate(noise_types):
            subset = df[(df['model'] == model) & (df['noise_dist'] == noise) & (df['comp_demo'] == True)]
            
            if len(subset) > 0:
                pos_fracs = []
                for idx, row in subset.iterrows():
                    delta = np.array(row['delta_norm'])
                    if len(delta) > 0:
                        pos_fracs.append(np.mean(delta > 0.05))
                
                if pos_fracs:
                    rate_matrix[i, j] = np.mean(pos_fracs)
    
    # Heatmap
    ax1 = axes[0, 0]
    im = ax1.imshow(rate_matrix, cmap='RdYlBu_r', aspect='auto', vmin=0, vmax=1)
    ax1.set_xticks(range(len(noise_types)))
    ax1.set_yticks(range(len(models)))
    ax1.set_xticklabels(noise_types)
    ax1.set_yticklabels([m.upper() for m in models])
    ax1.set_title('PE Correction Rate (Fraction δ > 0.05)')
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(noise_types)):
            text = ax1.text(j, i, f'{rate_matrix[i, j]:.3f}', 
                          ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax1, label='Correction Rate')
    
    # Effect sizes (if statistical analysis available)
    if stats_analysis and 'paired_t_tests' in stats_analysis:
        ax2 = axes[0, 1]
        
        # Extract effect sizes
        effect_sizes = []
        labels = []
        
        for key, result in stats_analysis['paired_t_tests'].items():
            if 'cohens_d' in result:
                effect_sizes.append(result['cohens_d'])
                labels.append(key.replace('_', ' ').title())
        
        if effect_sizes:
            bars = ax2.barh(range(len(effect_sizes)), effect_sizes, 
                          color=['red' if es > 0 else 'blue' for es in effect_sizes])
            ax2.set_yticks(range(len(labels)))
            ax2.set_yticklabels(labels)
            ax2.set_xlabel("Cohen's d (Effect Size)")
            ax2.set_title('Effect Sizes: PE vs PL')
            ax2.axvline(0, color='black', linestyle='-', alpha=0.5)
            ax2.axvline(0.2, color='gray', linestyle='--', alpha=0.5, label='Small')
            ax2.axvline(0.5, color='gray', linestyle='--', alpha=0.7, label='Medium')
            ax2.axvline(0.8, color='gray', linestyle='--', alpha=0.9, label='Large')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
    
    # ESS reliability analysis
    ax3 = axes[1, 0]
    
    ess_data = []
    model_labels = []
    
    for model in models:
        model_subset = df[df['model'] == model]
        all_ess = []
        
        for idx, row in model_subset.iterrows():
            ess = np.array(row['ess_rels'])
            if len(ess) > 0:
                all_ess.extend(ess)
        
        if all_ess:
            ess_data.append(all_ess)
            model_labels.append(model.upper())
    
    if ess_data:
        bp = ax3.boxplot(ess_data, labels=model_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral'][:len(bp['boxes'])]):
            patch.set_facecolor(color)
        
        ax3.axhline(0.1, color='red', linestyle='--', alpha=0.7, label='Target ESS ≥ 0.1')
        ax3.axhline(0.05, color='orange', linestyle=':', alpha=0.7, label='Acceptable ESS ≥ 0.05')
        ax3.set_ylabel('Effective Sample Size (ESS)')
        ax3.set_title('ESS Distribution by Model')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Gain-ridge correlation validation
    ax4 = axes[1, 1]
    
    all_corrs = []
    corr_labels = []
    
    for model in models:
        model_subset = df[df['model'] == model]
        model_corrs = []
        
        for idx, row in model_subset.iterrows():
            gains = np.array(row['comp_gains'])
            ridges = np.array(row['ridge_logvols'])
            
            if len(gains) > 1 and len(ridges) > 1 and len(gains) == len(ridges):
                try:
                    corr, _ = stats.spearmanr(gains, ridges)
                    if not np.isnan(corr):
                        model_corrs.append(corr)
                except:
                    pass
        
        if model_corrs:
            all_corrs.append(model_corrs)
            corr_labels.append(model.upper())
    
    if all_corrs:
        bp2 = ax4.boxplot(all_corrs, labels=corr_labels, patch_artist=True)
        for patch, color in zip(bp2['boxes'], ['lightgreen', 'lightyellow'][:len(bp2['boxes'])]):
            patch.set_facecolor(color)
        
        ax4.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax4.axhline(0.5, color='green', linestyle='--', alpha=0.7, label='Strong correlation')
        ax4.set_ylabel('Spearman Correlation (Gain vs Ridge)')
        ax4.set_title('Theory Validation: Gain-Ridge Correlation')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Figure 2: Statistical Significance and Validation', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_2_statistical_significance.png'))
    plt.savefig(os.path.join(output_dir, 'figure_2_statistical_significance.pdf'))
    plt.close()
    
    print("Generated Figure 2: Statistical significance")


def create_figure_3_computational_performance(performance_data: Optional[Dict], output_dir: str) -> None:
    """Figure 3: Computational performance benchmarks."""
    
    if not performance_data:
        print("No performance data available for Figure 3")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Runtime scaling
    if 'grid_scaling' in performance_data:
        grid_data = performance_data['grid_scaling']
        grid_sizes = [d['grid_size'] for d in grid_data]
        runtimes = [d['runtime_seconds'] for d in grid_data]
        
        ax1 = axes[0, 0]
        ax1.plot(grid_sizes, runtimes, 'bo-', linewidth=2, markersize=6)
        
        # Fit and plot trend
        z = np.polyfit(grid_sizes, runtimes, 2)
        p = np.poly1d(z)
        x_trend = np.linspace(min(grid_sizes), max(grid_sizes), 100)
        ax1.plot(x_trend, p(x_trend), 'r--', alpha=0.7, 
                label=f'Quadratic fit: {z[0]:.3f}x² + {z[1]:.2f}x + {z[2]:.1f}')
        
        ax1.set_xlabel('Grid Size (number of ω points)')
        ax1.set_ylabel('Runtime (seconds)')
        ax1.set_title('Computational Scaling: Grid Size')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Sample size scaling
    if 'sample_scaling' in performance_data:
        sample_data = performance_data['sample_scaling']
        sample_sizes = [d['sample_size'] for d in sample_data]
        runtimes = [d['runtime_seconds'] for d in sample_data]
        
        ax2 = axes[0, 1]
        ax2.plot(sample_sizes, runtimes, 'go-', linewidth=2, markersize=6)
        
        # Linear fit
        z = np.polyfit(sample_sizes, runtimes, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(min(sample_sizes), max(sample_sizes), 100)
        ax2.plot(x_trend, p(x_trend), 'r--', alpha=0.7, 
                label=f'Linear fit: {z[0]:.2e}x + {z[1]:.2f}')
        
        ax2.set_xlabel('Sample Size (importance samples)')
        ax2.set_ylabel('Runtime (seconds)')
        ax2.set_title('Computational Scaling: Sample Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # ESS optimization
    if 'ess_optimization' in performance_data:
        ess_data = performance_data['ess_optimization']
        scales = [d['proposal_scale'] for d in ess_data]
        ess_values = [d['ess'] for d in ess_data]
        
        ax3 = axes[1, 0]
        ax3.semilogx(scales, ess_values, 'mo-', linewidth=2, markersize=6)
        
        # Mark optimal
        optimal_scale = performance_data.get('optimal_scale', 1.0)
        ax3.axvline(optimal_scale, color='red', linestyle='--', alpha=0.7, 
                   label=f'Optimal: {optimal_scale:.1f}')
        
        ax3.set_xlabel('Proposal Scale Factor')
        ax3.set_ylabel('Effective Sample Size')
        ax3.set_title('ESS Optimization')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Performance comparison table (as text plot)
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Create performance summary table
    table_data = []
    if 'grid_scaling' in performance_data:
        max_grid = max(d['grid_size'] for d in performance_data['grid_scaling'])
        max_runtime = max(d['runtime_seconds'] for d in performance_data['grid_scaling'])
        table_data.append(['Grid Analysis', f'{max_grid} points', f'{max_runtime:.1f}s'])
    
    if 'sample_scaling' in performance_data:
        max_samples = max(d['sample_size'] for d in performance_data['sample_scaling'])
        max_sample_runtime = max(d['runtime_seconds'] for d in performance_data['sample_scaling'])
        table_data.append(['Importance Sampling', f'{max_samples:,} samples', f'{max_sample_runtime:.2f}s'])
    
    if 'ess_optimization' in performance_data:
        optimal_scale = performance_data.get('optimal_scale', 1.0)
        max_ess = performance_data.get('max_ess', 0.0)
        table_data.append(['ESS Optimization', f'{optimal_scale:.1f}× scale', f'{max_ess:.3f} ESS'])
    
    if table_data:
        table = ax4.table(cellText=table_data,
                         colLabels=['Component', 'Configuration', 'Performance'],
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0.3, 1, 0.4])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax4.text(0.5, 0.8, 'Performance Summary', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=12, fontweight='bold')
        
        # Add recommendations
        recommendations = [
            'Recommended Configuration:',
            '• Grid size: 20-25 points',
            '• Sample size: 2000-3000',
            f'• Proposal scale: {optimal_scale:.1f}×',
            '• Expected runtime: ~60-120s per analysis'
        ]
        
        ax4.text(0.5, 0.2, '\n'.join(recommendations), ha='center', va='center',
                transform=ax4.transAxes, fontsize=9, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.suptitle('Figure 3: Computational Performance Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_3_computational_performance.png'))
    plt.savefig(os.path.join(output_dir, 'figure_3_computational_performance.pdf'))
    plt.close()
    
    print("Generated Figure 3: Computational performance")


def create_figure_4_evidence_visualization(df: pd.DataFrame, output_dir: str) -> None:
    """Figure 4: Direct evidence visualization (compensation clouds and paths)."""
    
    # Find a good example with compensation
    comp_examples = df[(df['comp_demo'] == True) & (df['model'] == 'smib')]
    
    if len(comp_examples) == 0:
        print("No compensation examples found for Figure 4")
        return
    
    # Use first available example
    example = comp_examples.iloc[0]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # PL vs PE curves
    ax1 = axes[0, 0]
    omegas = np.array(example['omegas'])
    pl_vals = np.array(example['pl_vals'])
    pe_vals = np.array(example['pe_vals'])
    
    if len(omegas) > 0:
        pl_norm = pl_vals - np.max(pl_vals)
        pe_norm = pe_vals - np.max(pe_vals)
        
        ax1.plot(omegas, pl_norm, 'b-o', label='Profile Likelihood', linewidth=2, markersize=4)
        ax1.plot(omegas, pe_norm, 'r-s', label='Profile Evidence', linewidth=2, markersize=4)
        ax1.axvline(example['omega_true'], color='green', linestyle='--', 
                   label=f'True ω = {example["omega_true"]:.1f}', alpha=0.8)
        ax1.axhline(-0.5 * 3.841458820694124, color='gray', linestyle=':', 
                   alpha=0.7, label='95% threshold')
        
        ax1.set_xlabel('Parameter of Interest (ω)')
        ax1.set_ylabel('Normalized Log-Likelihood/Evidence')
        ax1.set_title(f'PE vs PL: {example["model"].upper()} (Seed {example["seed"]})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Delta curve
    ax2 = axes[0, 1]
    delta = np.array(example['delta_norm'])
    
    if len(delta) > 0 and len(omegas) > 0:
        ax2.plot(omegas, delta, 'purple', linewidth=2, marker='o', markersize=4)
        ax2.axhline(0, color='black', linestyle='-', alpha=0.5)
        ax2.axhline(0.05, color='green', linestyle='--', alpha=0.7, label='Correction threshold')
        ax2.axhline(-0.05, color='orange', linestyle='--', alpha=0.7, label='Overestimation threshold')
        
        # Highlight correction regions
        pos_mask = delta > 0.05
        neg_mask = delta < -0.05
        
        if np.any(pos_mask):
            ax2.fill_between(omegas, 0, delta, where=pos_mask, alpha=0.3, 
                           color='green', label='PE corrections')
        if np.any(neg_mask):
            ax2.fill_between(omegas, 0, delta, where=neg_mask, alpha=0.3, 
                           color='orange', label='PL overestimation')
        
        ax2.set_xlabel('Parameter of Interest (ω)')
        ax2.set_ylabel('Δ = PE_norm - PL_norm')
        ax2.set_title('Compensation Detection')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Gain vs Ridge correlation
    ax3 = axes[1, 0]
    gains = np.array(example['comp_gains'])
    ridges = np.array(example['ridge_logvols'])
    
    if len(gains) > 0 and len(ridges) > 0 and len(gains) == len(ridges):
        ax3.scatter(ridges, gains, c=omegas if len(omegas) == len(gains) else 'blue', 
                   cmap='viridis', alpha=0.7, s=50)
        
        # Fit line
        if len(gains) > 2:
            z = np.polyfit(ridges, gains, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(ridges.min(), ridges.max(), 100)
            ax3.plot(x_trend, p(x_trend), 'r--', alpha=0.7, 
                    label=f'Linear fit: slope = {z[0]:.2f}')
            
            # Correlation
            corr, p_val = stats.spearmanr(gains, ridges)
            ax3.text(0.05, 0.95, f'ρ = {corr:.3f}\np = {p_val:.4f}', 
                    transform=ax3.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        if len(omegas) == len(gains):
            plt.colorbar(ax3.collections[0], ax=ax3, label='ω')
        
        ax3.set_xlabel('Ridge Log-Volume')
        ax3.set_ylabel('Compensation Gain (Laplace - PL)')
        ax3.set_title('Theory Validation: Gain vs Ridge Volume')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # ESS quality assessment
    ax4 = axes[1, 1]
    ess = np.array(example['ess_rels'])
    
    if len(ess) > 0 and len(omegas) > 0:
        bars = ax4.bar(range(len(ess)), ess, 
                      color=['green' if e >= 0.1 else 'orange' if e >= 0.05 else 'red' for e in ess])
        
        ax4.axhline(0.1, color='red', linestyle='--', alpha=0.7, label='Target ESS ≥ 0.1')
        ax4.axhline(0.05, color='orange', linestyle=':', alpha=0.7, label='Acceptable ESS ≥ 0.05')
        
        # Add omega labels (subset for readability)
        step = max(1, len(omegas) // 8)
        ax4.set_xticks(range(0, len(omegas), step))
        ax4.set_xticklabels([f'{omegas[i]:.2f}' for i in range(0, len(omegas), step)], 
                          rotation=45)
        
        ax4.set_xlabel('Parameter Value (ω)')
        ax4.set_ylabel('Effective Sample Size')
        ax4.set_title(f'Sampling Quality (Mean ESS: {np.mean(ess):.3f})')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.suptitle('Figure 4: Direct Evidence and Quality Assessment', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'figure_4_evidence_visualization.png'))
    plt.savefig(os.path.join(output_dir, 'figure_4_evidence_visualization.pdf'))
    plt.close()
    
    print("Generated Figure 4: Evidence visualization")


def generate_figure_captions(output_dir: str) -> None:
    """Generate LaTeX figure captions for paper."""
    
    captions = {
        'figure_1': r"""
\caption{Multi-model comparison of Profile Evidence (PE) versus Profile Likelihood (PL). 
Left panels show normalized curves with confidence bands (shaded regions represent ±1 standard deviation across seeds). 
Right panels show compensation detection via $\Delta = \text{PE}_{\text{norm}} - \text{PL}_{\text{norm}}$. 
Positive $\Delta$ indicates PE correction of PL underestimation; negative $\Delta$ indicates PL overestimation detection. 
Dashed lines at $\pm 0.05$ represent significance thresholds for compensation detection.}
        """,
        
        'figure_2': r"""
\caption{Statistical significance and validation analysis. 
(a) Heatmap of PE correction rates by model and noise distribution. 
(b) Effect sizes (Cohen's $d$) for paired comparisons of PE vs PL. 
(c) Effective Sample Size (ESS) distributions demonstrating sampling reliability ($\text{ESS} \geq 0.1$ target). 
(d) Gain-ridge correlation validation supporting theoretical predictions of compensation mechanism.}
        """,
        
        'figure_3': r"""
\caption{Computational performance benchmarks. 
(a) Runtime scaling with grid size showing quadratic complexity due to Hessian computations. 
(b) Linear scaling with importance sampling size. 
(c) ESS optimization demonstrating optimal proposal scale selection. 
(d) Performance summary with recommended configurations for production use.}
        """,
        
        'figure_4': r"""
\caption{Direct evidence visualization for compensation detection. 
(a) Normalized PE vs PL curves with 95\% confidence threshold. 
(b) Compensation detection curve highlighting correction and overestimation regions. 
(c) Theoretical validation through gain-ridge volume correlation with Spearman coefficient. 
(d) Sampling quality assessment showing ESS across parameter grid with reliability indicators.}
        """
    }
    
    # Save captions to file
    with open(os.path.join(output_dir, 'figure_captions.tex'), 'w') as f:
        f.write("% Figure captions for PE vs PL paper\n\n")
        for fig_name, caption in captions.items():
            f.write(f"% {fig_name.upper()}\n")
            f.write(caption.strip() + "\n\n")
    
    print("Generated LaTeX figure captions")


def main():
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'outputs'
    output_dir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(results_dir, 'paper_figures')
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Generating publication figures...")
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load data
    print("Loading results data...")
    df = load_results_data(results_dir)
    print(f"Loaded {len(df)} result files")
    
    if len(df) == 0:
        print("No results found!")
        return
    
    # Load statistical analysis if available
    stats_analysis = load_statistical_analysis(results_dir)
    if stats_analysis:
        print("Loaded statistical analysis results")
    
    # Load performance data if available
    performance_files = glob.glob(os.path.join(results_dir, '**/benchmark_results.pkl'), recursive=True)
    performance_data = None
    if performance_files:
        latest_perf = max(performance_files, key=os.path.getmtime)
        with open(latest_perf, 'rb') as f:
            performance_data = pickle.load(f)
        print("Loaded performance benchmark results")
    
    # Generate figures
    print("Creating figures...")
    
    create_figure_1_multi_model_comparison(df, output_dir)
    create_figure_2_statistical_significance(df, stats_analysis, output_dir)
    create_figure_3_computational_performance(performance_data, output_dir)
    create_figure_4_evidence_visualization(df, output_dir)
    
    # Generate captions
    generate_figure_captions(output_dir)
    
    # Save metadata
    metadata = {
        'generated': datetime.now().isoformat(),
        'results_dir': results_dir,
        'n_results': len(df),
        'models': sorted(df['model'].unique()),
        'has_stats_analysis': stats_analysis is not None,
        'has_performance_data': performance_data is not None
    }
    
    with open(os.path.join(output_dir, 'figure_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nAll figures generated successfully!")
    print(f"Output directory: {output_dir}")
    print("Files created:")
    for file in sorted(os.listdir(output_dir)):
        print(f"  - {file}")


if __name__ == '__main__':
    main()
