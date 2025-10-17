#!/usr/bin/env python3
"""
Section IV: Numerical Results and Discussion - Complete Simulation Suite
=======================================================================

IEEE Transactions Paper - OISL-ISAC Performance Limits
Main driver script for generating all figures and validation experiments.

Usage:
    python run_section_iv.py --figure all           # Generate all figures
    python run_section_iv.py --figure 1             # Generate specific figure
    python run_section_iv.py --validation           # Run Monte Carlo validation
    python run_section_iv.py --ergodic              # Run Ergodic vs Outage analysis
"""
# 在 run_section_iv.py 顶部
import multiprocessing as mp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import sys
import argparse
from pathlib import Path
from tqdm import tqdm
import warnings
from scipy.special import gammaln
from scipy.optimize import minimize_scalar
from scipy.stats import poisson
import time
from datetime import datetime

# Import core simulation functions
sys.path.append('.')
from main_simulation import (
    capacity_lb, fim_pilot, generate_dither_sequence,
    poisson_entropy, setup_ieee_style, SPEED_OF_LIGHT
)

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

def load_config(config_path='config.yaml'):
    """Load simulation parameters from YAML config"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(base_dir='section_iv_results'):
    """Create organized output directory structure"""
    dirs = {
        'base': base_dir,
        'figures': f'{base_dir}/figures',
        'data': f'{base_dir}/data',
        'validation': f'{base_dir}/validation',
        'appendix': f'{base_dir}/appendix'
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def set_reproducibility(seed=42):
    """Set global random seed for reproducibility"""
    np.random.seed(seed)
    print(f"🔒 Random seed set to: {seed}")


# ============================================================================
# FIGURE 1: CAPACITY vs BACKGROUND & PLATFORM CONSTRAINTS
# ============================================================================

def generate_figure_1(config, dirs):
    """
    Generate Fig_1_Capacity_Platform.pdf
    Shows capacity vs background with hardware platform constraints
    """
    print("\n" + "=" * 60)
    print("📊 FIGURE 1: Capacity vs Background & Platform")
    print("=" * 60)

    # Setup plotting
    colors = setup_ieee_style()

    # Parameters from config
    lambda_b_range = np.logspace(-2, 2, 80)  # Background range
    signal_budgets = config['simulation']['signal_budgets']  # [0.5, 2, 10, 50]
    hardware_configs = config['hardware_platforms']

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for idx, S_bar in enumerate(signal_budgets):
        ax = axes[idx]

        for hw_name, hw_config in hardware_configs.items():
            tau_d = hw_config['dead_time']
            M_pixels = hw_config['parallel_pixels']
            dt = hw_config['slot_duration']

            # Compute effective S_max
            if tau_d > 0:
                S_max_eff = min(hw_config['peak_power'], (dt / tau_d) * M_pixels)
            else:
                S_max_eff = hw_config['peak_power']

            capacities = []

            for lambda_b in lambda_b_range:
                C_lb, _ = capacity_lb(S_bar, S_max_eff, lambda_b, dt, tau_d, M_pixels)
                capacities.append(C_lb)

            # Plot with different line styles
            linestyle = '--' if 'long' in hw_name.lower() else '-'
            linewidth = 2.5 if 'short' in hw_name.lower() else 1.8

            ax.semilogx(lambda_b_range, capacities,
                        linestyle=linestyle, linewidth=linewidth,
                        label=f'{hw_name} (τ={tau_d * 1e9:.0f}ns, M={M_pixels})')

        # Mark background regimes
        ax.axvline(x=0.01, color='blue', alpha=0.3, linestyle=':', label='Zodiacal' if idx == 0 else "")
        ax.axvline(x=1.0, color='orange', alpha=0.3, linestyle=':', label='Earthshine' if idx == 0 else "")
        ax.axvline(x=10.0, color='red', alpha=0.3, linestyle=':', label='Stray Light' if idx == 0 else "")

        ax.set_xlabel('Background λ_b [photons/slot]')
        ax.set_ylabel('Capacity C_LB [bits/slot]')
        ax.set_title(f'S̄ = {S_bar} photons/slot')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7)

    plt.tight_layout()

    # Save figure
    output_path = f"{dirs['figures']}/Fig_1_Capacity_Platform"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}.pdf")

    # Save numerical data
    data_dict = {
        'lambda_b_range': lambda_b_range,
        'signal_budgets': signal_budgets,
        'hardware_configs': hardware_configs
    }
    np.savez(f"{dirs['data']}/fig1_data.npz", **data_dict)


# ============================================================================
# FIGURE 2: FIM HEATMAP COMPARISON
# ============================================================================
def _figure2_scenario_worker(args):
    """Worker for single scenario in Figure 2"""
    (scenario_name, scenario, params, alpha_range, rho_range, dither_seq) = args

    params_sim = params.copy()
    params_sim['r_b'] = scenario['r_b']

    mse_grid = np.zeros((len(rho_range), len(alpha_range)))
    valid_count = 0

    for i, rho in enumerate(rho_range):
        for j, alpha in enumerate(alpha_range):
            try:
                I_pilot = fim_pilot(alpha, rho, params['Sbar'], params['N'],
                                    params['dt'], params_sim, dither_seq,
                                    params.get('tau_d'), None, params['M_pixels'])

                J = I_pilot + params['J_P'] + 1e-12 * np.eye(4)

                if np.linalg.cond(J) < 1e30:
                    J_inv = np.linalg.inv(J)
                    W = np.diag([1.0, 1.0, 0.0, 0.0])
                    mse_grid[i, j] = np.trace(W @ J_inv)
                    valid_count += 1
                else:
                    mse_grid[i, j] = np.nan
            except:
                mse_grid[i, j] = np.nan

    return (scenario_name, scenario, mse_grid, valid_count)


def generate_figure_2(config, dirs, n_workers=2):
    """
    ** PARALLEL VERSION - 2 scenarios in parallel **
    """
    print("\n" + "=" * 60)
    print("📊 FIGURE 2: FIM Heatmap Comparison (PARALLEL)")
    print("=" * 60)

    print(f"🚀 Computing 2 scenarios in parallel")

    setup_ieee_style()

    # Parameters
    params = config['system_parameters']
    alpha_range = np.linspace(0.05, 0.95, 25)
    rho_range = np.linspace(0.05, 0.95, 25)

    scenarios = {
        'Low Background (Zodiacal)': {'r_b': 0.01, 'color': 'viridis'},
        'High Background (Stray Light)': {'r_b': 10.0, 'color': 'plasma'}
    }

    # Generate dither
    max_pilots = int(0.9 * params['N'])
    dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

    # Prepare worker arguments
    worker_args = [
        (name, scenario, params, alpha_range, rho_range, dither_seq)
        for name, scenario in scenarios.items()
    ]

    # Execute in parallel
    print("🔄 Computing FIM grids in parallel...")
    with mp.Pool(min(2, n_workers)) as pool:
        results = pool.map(_figure2_scenario_worker, worker_args)

    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (scenario_name, scenario, mse_grid, valid_count) in enumerate(results):
        ax = axes[idx]

        print(f"  {scenario_name}: {valid_count}/{mse_grid.size} valid points")

        # Plot heatmap (保持原有绘图代码)
        valid_mask = np.isfinite(mse_grid) & (mse_grid > 0)
        if valid_mask.sum() > 0:
            vmin = np.percentile(mse_grid[valid_mask], 5)
            vmax = np.percentile(mse_grid[valid_mask], 95)

            log_mse = np.full_like(mse_grid, np.nan)
            log_mse[valid_mask] = np.log10(mse_grid[valid_mask])

            Alpha, Rho = np.meshgrid(alpha_range, rho_range)

            im = ax.pcolormesh(Alpha, Rho, log_mse, shading='auto',
                               cmap=scenario['color'], vmin=np.log10(vmin), vmax=np.log10(vmax))

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('log₁₀(MSE μx,μy) [rad²]', rotation=270, labelpad=15)

        ax.set_xlabel('Time Allocation α')
        ax.set_ylabel('Photon Allocation ρ')
        ax.set_title(scenario_name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = f"{dirs['figures']}/Fig_2_FIM_Heatmap_Comparison"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}.pdf")

# ============================================================================
# FIGURE 3: RATE-MSE PARETO BOUNDARY
# ============================================================================

# ============================================================================
# WORKER FUNCTION FOR FIGURE 3 PARETO (新增)
# ============================================================================

def _figure3_pareto_point_worker(args):
    """
    Worker for Figure 3 Pareto boundary computation

    Similar to main_simulation.py but adapted for run_section_iv.py
    """
    (D_max, alpha_search, rho_search, params, scenario_r_b,
     dither_seq, Smax_eff) = args

    max_rate = 0.0
    best_alpha, best_rho = 0, 0

    for alpha in alpha_search:
        for rho in rho_search:
            try:
                # Check feasibility
                S_pilot = rho * params['Sbar'] / alpha
                S_data = (1 - rho) * params['Sbar'] / (1 - alpha)

                if S_pilot > Smax_eff or S_data > Smax_eff:
                    continue

                # Compute FIM
                params_sim = params.copy()
                params_sim['r_b'] = scenario_r_b

                I_pilot = fim_pilot(alpha, rho, params['Sbar'], params['N'],
                                    params['dt'], params_sim, dither_seq,
                                    params.get('tau_d'), None, params['M_pixels'])

                J = I_pilot + params['J_P'] + 1e-12 * np.eye(4)

                if np.linalg.cond(J) > 1e30:
                    continue

                J_inv = np.linalg.inv(J)
                W = np.diag([1.0, 1.0, 0.0, 0.0])
                mse_current = np.trace(W @ J_inv)

                if mse_current > D_max:
                    continue

                # Compute achievable rate
                C_data, _ = capacity_lb(S_data, Smax_eff, scenario_r_b,
                                        params['dt'], params.get('tau_d'), params['M_pixels'])
                rate = (1 - alpha) * C_data

                if rate > max_rate:
                    max_rate = rate
                    best_alpha, best_rho = alpha, rho
            except:
                continue

    return (max_rate, D_max, best_alpha, best_rho) if max_rate > 0 else None


def generate_figure_3(config, dirs, n_workers=None):
    """
    Generate Fig_3_Rate_MSE_Boundary.pdf
    Shows Rate-MSE Pareto boundaries for different background scenarios

    ** PARALLEL VERSION **
    """
    print("\n" + "=" * 60)
    print("📊 FIGURE 3: Rate-MSE Pareto Boundary (PARALLEL)")
    print("=" * 60)

    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    n_workers = max(1, n_workers)

    print(f"🚀 Using {n_workers} parallel worker(s)")

    colors = setup_ieee_style()

    # Parameters
    params = config['system_parameters']

    # Background scenarios
    scenarios = {
        'Low (Zodiacal)': {'r_b': 0.01, 'color': colors['zodiacal']},
        'Medium (Earthshine)': {'r_b': 1.0, 'color': colors['earthshine']},
        'High (Stray Light)': {'r_b': 10.0, 'color': colors['stray_light']}
    }

    # ========================================================================
    # PHASE 1: Adaptive MSE targets based on achievable range
    # ========================================================================

    print("🔍 Probing achievable MSE range...")
    alpha_probe = np.linspace(0.1, 0.9, 8)
    rho_probe = np.linspace(0.1, 0.9, 8)

    # Use medium scenario for probing
    params_probe = params.copy()
    params_probe['r_b'] = 1.0

    max_pilots = int(0.9 * params['N'])
    dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

    mse_samples = []

    for alpha in alpha_probe:
        for rho in rho_probe:
            try:
                I_pilot = fim_pilot(alpha, rho, params['Sbar'], params['N'],
                                    params['dt'], params_probe, dither_seq,
                                    params.get('tau_d'), None, params['M_pixels'])
                J = I_pilot + params['J_P'] + 1e-12 * np.eye(4)
                if np.linalg.cond(J) < 1e30:
                    W = np.diag([1.0, 1.0, 0.0, 0.0])
                    mse = np.trace(W @ np.linalg.inv(J))
                    if np.isfinite(mse) and mse > 0:
                        mse_samples.append(mse)
            except:
                pass

    if len(mse_samples) > 10:
        mmin, mmax = np.percentile(mse_samples, [5, 95])
        D_targets = np.logspace(np.log10(mmin * 0.8), np.log10(mmax * 1.2), 15)
    else:
        print("⚠️  Using fallback MSE range")
        D_targets = np.logspace(-12, -6, 15)

    print(f"✅ MSE targets: [{D_targets[0]:.2e}, {D_targets[-1]:.2e}]")

    # ========================================================================
    # PHASE 2: Compute Pareto boundaries (PARALLEL)
    # ========================================================================

    alpha_search = np.linspace(0.05, 0.95, 20)
    rho_search = np.linspace(0.05, 0.95, 20)

    # Effective Smax
    if params.get('tau_d'):
        Smax_eff = min(params['Smax'],
                       (params['dt'] / params['tau_d']) * params['M_pixels'])
    else:
        Smax_eff = params['Smax']

    pareto_results = {}

    for scenario_name, scenario in scenarios.items():
        print(f"\n{'=' * 60}")
        print(f"🔄 Computing Pareto boundary: {scenario_name}")
        print(f"{'=' * 60}")

        params_sim = params.copy()
        params_sim['r_b'] = scenario['r_b']

        # Prepare worker arguments
        worker_args = [
            (D_max, alpha_search, rho_search, params, scenario['r_b'],
             dither_seq, Smax_eff)
            for D_max in D_targets
        ]

        # Execute in parallel
        if n_workers > 1:
            print(f"🔄 Parallel computation with {n_workers} workers...")
            with mp.Pool(n_workers) as pool:
                results = list(tqdm(
                    pool.imap(_figure3_pareto_point_worker, worker_args),
                    total=len(D_targets),
                    desc=f"  {scenario_name}"
                ))
        else:
            print(f"🔄 Sequential computation (single worker)...")
            results = [
                _figure3_pareto_point_worker(args)
                for args in tqdm(worker_args, desc=f"  {scenario_name}")
            ]

        # Filter valid results
        pareto_points = [r for r in results if r is not None]
        pareto_results[scenario_name] = [
            (rate, D_max) for (rate, D_max, _, _) in pareto_points
        ]

        print(f"✅ Found {len(pareto_points)}/{len(D_targets)} valid Pareto points")

    # ========================================================================
    # PHASE 3: Plot results
    # ========================================================================

    print(f"\n{'=' * 60}")
    print("📊 Generating plot...")
    print(f"{'=' * 60}")

    fig, ax = plt.subplots(figsize=(8, 6))

    for scenario_name, scenario in scenarios.items():
        points = pareto_results[scenario_name]
        if len(points) > 0:
            rates = [p[0] for p in points]
            mses = [p[1] for p in points]

            # Sort by MSE for proper plotting
            sorted_pairs = sorted(zip(mses, rates))
            mses_sorted = [p[0] for p in sorted_pairs]
            rates_sorted = [p[1] for p in sorted_pairs]

            ax.loglog(mses_sorted, rates_sorted, 'o-',
                      color=scenario['color'], linewidth=2.5, markersize=6,
                      label=scenario_name)

            # Annotate a few representative points
            if len(mses_sorted) > 2:
                mid_idx = len(mses_sorted) // 2
                # Note: We don't have alphas/rhos here, skip annotation
                # ax.annotate(...)

        else:
            print(f"⚠️  No valid points for {scenario_name}")

    ax.set_xlabel('MSE (μx, μy) [rad²]', fontweight='bold')
    ax.set_ylabel('Rate [bits/slot]', fontweight='bold')
    ax.set_title(f'Rate-MSE Pareto Boundary\n(S̄={params["Sbar"]}, M={params["M_pixels"]})',
                 fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    # Save figure
    output_path = f"{dirs['figures']}/Fig_3_Rate_MSE_Boundary"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}.pdf")


# ============================================================================
# FIGURE 4: SUN AVOIDANCE ANGLE DESIGN LAW
# ============================================================================
# ============================================================================
# FIGURE 4 DIAGNOSTICS (新增)
# ============================================================================

def diagnose_figure_4(config, dirs):
    """
    Complete diagnostic for Figure 4 Design Law

    This function:
    1. Computes the same grids as generate_figure_4
    2. Analyzes data ranges
    3. Identifies why contours might not appear
    4. Suggests fixes
    """
    print("\n" + "=" * 80)
    print("🔬 FIGURE 4 COMPREHENSIVE DIAGNOSTICS")
    print("=" * 80)

    setup_ieee_style()

    # Same parameters as generate_figure_4
    sun_angles = np.linspace(10, 180, 50)
    fov_range = np.linspace(20, 500, 40)

    # Original background model from generate_figure_4
    def background_rate(sun_angle_deg, fov_urad):
        """Original background model"""
        sun_angle_rad = np.radians(sun_angle_deg)
        fov_rad = fov_urad * 1e-6

        log_pst = -4.0 - 3.0 * np.log10(sun_angle_deg / 10.0)
        log_pst = np.clip(log_pst, -10, -4)
        pst = 10 ** log_pst

        lambda_b = pst * (fov_rad / 100e-6) ** 2 * 1000

        return lambda_b

    # Create grids
    Sun_grid, FoV_grid = np.meshgrid(sun_angles, fov_range)
    Background_grid = np.zeros_like(Sun_grid)
    Capacity_grid = np.zeros_like(Sun_grid)

    # System parameters
    params = config['system_parameters']
    S_bar = params['Sbar']
    S_max = params['Smax']
    dt = params['dt']
    tau_d = params.get('tau_d')
    M_pixels = params['M_pixels']

    print(f"\n📋 System Parameters:")
    print(f"   S̄ = {S_bar} photons/slot")
    print(f"   S_max = {S_max} photons/slot")
    print(f"   dt = {dt * 1e6:.2f} μs")
    print(f"   M_pixels = {M_pixels}")

    # Compute grids
    print(f"\n🔄 Computing grids ({len(fov_range)} × {len(sun_angles)} = {len(fov_range) * len(sun_angles)} points)...")

    for i in tqdm(range(len(fov_range)), desc="Computing"):
        for j in range(len(sun_angles)):
            lambda_b = background_rate(Sun_grid[i, j], FoV_grid[i, j])
            Background_grid[i, j] = lambda_b

            C_lb, _ = capacity_lb(S_bar, S_max, lambda_b, dt, tau_d, M_pixels)
            Capacity_grid[i, j] = C_lb

    # ========================================================================
    # ANALYSIS 1: Background Grid
    # ========================================================================

    print("\n" + "=" * 80)
    print("📊 ANALYSIS 1: Background Grid (λ_b)")
    print("=" * 80)

    bg_min, bg_max = np.min(Background_grid), np.max(Background_grid)
    bg_mean, bg_median = np.mean(Background_grid), np.median(Background_grid)

    print(f"\n✓ Basic Statistics:")
    print(f"   Min:    {bg_min:.6e} photons/slot")
    print(f"   Max:    {bg_max:.6e} photons/slot")
    print(f"   Mean:   {bg_mean:.6e} photons/slot")
    print(f"   Median: {bg_median:.6e} photons/slot")
    print(f"   Range:  {bg_max / bg_min:.2e}x variation")

    print(f"\n✓ Log10 Scale:")
    print(f"   log10(min):  {np.log10(bg_min):.2f}")
    print(f"   log10(max):  {np.log10(bg_max):.2f}")
    print(f"   log10(span): {np.log10(bg_max) - np.log10(bg_min):.2f} decades")

    # Check for problematic values
    n_zero = np.sum(Background_grid == 0)
    n_nan = np.sum(np.isnan(Background_grid))
    n_inf = np.sum(np.isinf(Background_grid))

    print(f"\n✓ Data Quality:")
    print(f"   Zeros: {n_zero}/{Background_grid.size} ({n_zero / Background_grid.size * 100:.2f}%)")
    print(f"   NaNs:  {n_nan}/{Background_grid.size} ({n_nan / Background_grid.size * 100:.2f}%)")
    print(f"   Infs:  {n_inf}/{Background_grid.size} ({n_inf / Background_grid.size * 100:.2f}%)")

    if n_zero + n_nan + n_inf > 0:
        print(f"   ⚠️  WARNING: Found {n_zero + n_nan + n_inf} problematic values!")
    else:
        print(f"   ✅ All background values are finite and positive")

    # Sample some specific points
    print(f"\n✓ Sample Points:")
    test_points = [
        (10, 50, "Near sun, typical FoV"),
        (30, 50, "Min avoidance, typical FoV"),
        (90, 50, "90°, typical FoV"),
        (180, 50, "Deep space, typical FoV"),
        (90, 200, "90°, wide FoV"),
    ]

    for sun_deg, fov_urad, desc in test_points:
        lambda_b = background_rate(sun_deg, fov_urad)
        print(f"   ({sun_deg:3d}°, {fov_urad:3d}μrad) = {lambda_b:.4e} | {desc}")

    # ========================================================================
    # ANALYSIS 2: Capacity Grid
    # ========================================================================

    print("\n" + "=" * 80)
    print("📊 ANALYSIS 2: Capacity Grid")
    print("=" * 80)

    c_min, c_max = np.min(Capacity_grid), np.max(Capacity_grid)
    c_mean, c_median = np.mean(Capacity_grid), np.median(Capacity_grid)
    c_std = np.std(Capacity_grid)

    print(f"\n✓ Basic Statistics:")
    print(f"   Min:    {c_min:.6f} bits/slot")
    print(f"   Max:    {c_max:.6f} bits/slot")
    print(f"   Mean:   {c_mean:.6f} bits/slot")
    print(f"   Median: {c_median:.6f} bits/slot")
    print(f"   Std:    {c_std:.6f} bits/slot")
    print(f"   Range:  {c_max - c_min:.6f} bits/slot")

    # Check for problematic values
    n_zero = np.sum(Capacity_grid == 0)
    n_nan = np.sum(np.isnan(Capacity_grid))
    n_inf = np.sum(np.isinf(Capacity_grid))

    print(f"\n✓ Data Quality:")
    print(f"   Zeros: {n_zero}/{Capacity_grid.size} ({n_zero / Capacity_grid.size * 100:.2f}%)")
    print(f"   NaNs:  {n_nan}/{Capacity_grid.size} ({n_nan / Capacity_grid.size * 100:.2f}%)")
    print(f"   Infs:  {n_inf}/{Capacity_grid.size} ({n_inf / Capacity_grid.size * 100:.2f}%)")

    if n_zero + n_nan + n_inf > 0:
        print(f"   ⚠️  WARNING: Found {n_zero + n_nan + n_inf} problematic values!")
    else:
        print(f"   ✅ All capacity values are finite")

    # Distribution analysis
    print(f"\n✓ Distribution (percentiles):")
    percentiles = [0, 10, 25, 50, 75, 90, 100]
    for p in percentiles:
        val = np.percentile(Capacity_grid, p)
        print(f"   P{p:3d}: {val:.4f} bits/slot")

    # ========================================================================
    # ANALYSIS 3: Contour Level Compatibility
    # ========================================================================

    print("\n" + "=" * 80)
    print("📊 ANALYSIS 3: Contour Level Check (ROOT CAUSE)")
    print("=" * 80)

    # Original requested levels
    requested_levels = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]

    print(f"\n✓ Requested Contour Levels:")
    print(f"   {requested_levels}")

    print(f"\n✓ Capacity Range:")
    print(f"   [{c_min:.4f}, {c_max:.4f}] bits/slot")

    # Check which levels are valid
    valid_levels = [l for l in requested_levels if c_min <= l <= c_max]
    invalid_below = [l for l in requested_levels if l < c_min]
    invalid_above = [l for l in requested_levels if l > c_max]

    print(f"\n✓ Level Validation:")
    print(f"   Valid (within range):   {valid_levels if valid_levels else 'NONE! ❌'}")
    print(f"   Too low (< {c_min:.4f}):  {invalid_below}")
    print(f"   Too high (> {c_max:.4f}): {invalid_above}")

    # Diagnosis
    print(f"\n" + "=" * 80)
    print("🔍 DIAGNOSIS")
    print("=" * 80)

    if len(valid_levels) == 0:
        print(f"\n❌ PROBLEM IDENTIFIED: No valid contour levels!")
        print(f"\n   Root cause:")
        if c_max < requested_levels[0]:
            print(f"   → All capacity values ({c_max:.4f} max) are BELOW")
            print(f"     the minimum requested level ({requested_levels[0]:.4f})")
            print(f"\n   Why this happens:")
            print(f"   → Background is too high relative to signal budget")
            print(f"   → S̄ = {S_bar} may be insufficient for these backgrounds")
            print(f"   → Capacity ~ log(1 + S̄/λ_b) is very low when λ_b is large")
        elif c_min > requested_levels[-1]:
            print(f"   → All capacity values ({c_min:.4f} min) are ABOVE")
            print(f"     the maximum requested level ({requested_levels[-1]:.4f})")
            print(f"\n   Why this happens:")
            print(f"   → Background is very low, capacity is high everywhere")

        print(f"\n   Consequence:")
        print(f"   → matplotlib.contour() finds NO isoline to draw")
        print(f"   → Result: Only reference lines appear, no capacity contours")

    elif len(valid_levels) < 3:
        print(f"\n⚠️  WARNING: Only {len(valid_levels)} valid levels")
        print(f"   → Contours will be sparse")
        print(f"   → Consider adjusting levels for better visualization")
    else:
        print(f"\n✅ GOOD: {len(valid_levels)} valid contour levels")
        print(f"   → Contours should appear correctly")

    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================

    print(f"\n" + "=" * 80)
    print("💡 RECOMMENDATIONS")
    print("=" * 80)

    if len(valid_levels) == 0:
        print(f"\n🔧 SOLUTION: Use adaptive contour levels")

        # Generate adaptive levels
        if c_max > c_min + 0.01:
            adaptive_levels = np.linspace(c_min + 0.05 * (c_max - c_min),
                                          c_max - 0.05 * (c_max - c_min), 6)
            adaptive_levels = np.round(adaptive_levels, 3)

            print(f"\n   Suggested adaptive levels:")
            print(f"   {list(adaptive_levels)}")

            print(f"\n   Replace in generate_figure_4():")
            print(f"   OLD: capacity_levels = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]")
            print(f"   NEW: capacity_levels = {list(adaptive_levels)}")
        else:
            print(f"\n   ⚠️  Capacity range is too narrow ({c_max - c_min:.6f})")
            print(f"   → Consider increasing S̄ or improving background model")

    elif len(valid_levels) < len(requested_levels):
        print(f"\n🔧 PARTIAL FIX: Extend level range")

        extended_levels = np.linspace(c_min * 0.95, c_max * 0.95, 8)
        extended_levels = np.round(extended_levels, 3)

        print(f"\n   Suggested extended levels:")
        print(f"   {list(extended_levels)}")

    else:
        print(f"\n✅ Current levels are adequate")
        print(f"   → If contours still don't appear, check matplotlib version")
        print(f"   → Try: pip install --upgrade matplotlib")

    # ========================================================================
    # TEST PLOT
    # ========================================================================

    print(f"\n" + "=" * 80)
    print("📊 GENERATING TEST PLOT")
    print("=" * 80)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Background heatmap
    ax1 = axes[0]
    im1 = ax1.pcolormesh(Sun_grid, FoV_grid, np.log10(Background_grid + 1e-10),
                         shading='auto', cmap='YlOrRd', alpha=0.8)
    plt.colorbar(im1, ax=ax1, label='log₁₀(λ_b)')
    ax1.set_xlabel('Sun Angle [deg]')
    ax1.set_ylabel('FoV [μrad]')
    ax1.set_title('Background (should see smooth gradient)')
    ax1.grid(True, alpha=0.3)

    # Right: Capacity with adaptive contours
    ax2 = axes[1]
    im2 = ax2.pcolormesh(Sun_grid, FoV_grid, Capacity_grid,
                         shading='auto', cmap='viridis', alpha=0.6)
    plt.colorbar(im2, ax=ax2, label='Capacity [bits/slot]')

    # Try to draw contours with adaptive levels
    if c_max > c_min + 0.01:
        test_levels = np.percentile(Capacity_grid, [20, 35, 50, 65, 80])
        try:
            contours = ax2.contour(Sun_grid, FoV_grid, Capacity_grid,
                                   levels=test_levels, colors='red', linewidths=2)
            ax2.clabel(contours, inline=True, fontsize=9)
            print(f"\n✅ Test contours drawn successfully!")
            print(f"   Levels used: {test_levels}")
        except Exception as e:
            print(f"\n❌ Test contour failed: {e}")

    ax2.set_xlabel('Sun Angle [deg]')
    ax2.set_ylabel('FoV [μrad]')
    ax2.set_title('Capacity (contours should be visible)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save diagnostic plot
    diag_path = f"{dirs['figures']}/Fig_4_Diagnostic"
    plt.savefig(f"{diag_path}.png", dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Diagnostic plot saved: {diag_path}.png")
    print(f"   → Check if contours appear in the right panel")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print(f"\n" + "=" * 80)
    print("📋 DIAGNOSTIC SUMMARY")
    print("=" * 80)

    print(f"\n1. Background Grid:  {'✅ OK' if n_nan + n_inf == 0 else '❌ HAS ISSUES'}")
    print(f"2. Capacity Grid:    {'✅ OK' if np.all(np.isfinite(Capacity_grid)) else '❌ HAS ISSUES'}")
    print(f"3. Contour Levels:   {'✅ OK' if len(valid_levels) >= 3 else '❌ NEEDS FIX'}")

    if len(valid_levels) == 0:
        print(f"\n🎯 ACTION REQUIRED:")
        print(f"   Replace generate_figure_4() with the improved version")
        print(f"   (adaptive contour levels)")
    else:
        print(f"\n🎯 STATUS:")
        print(f"   Diagnostics complete. Review test plot for visual confirmation.")

    print(f"\n" + "=" * 80)

    return {
        'background_grid': Background_grid,
        'capacity_grid': Capacity_grid,
        'valid_levels': valid_levels,
        'adaptive_levels': np.percentile(Capacity_grid, [20, 35, 50, 65, 80]) if c_max > c_min else None
    }


def generate_figure_4(config, dirs):
    """
    Generate Fig_4_Design_Law.pdf
    Shows sun avoidance angle design law with background heatmap and capacity contours

    ** CLEAN VERSION - Single contour level generation **
    """
    print("\n" + "=" * 60)
    print("📊 FIGURE 4: Sun Avoidance Design Law")
    print("=" * 60)

    setup_ieee_style()

    # Grid parameters
    sun_angles = np.linspace(10, 180, 50)
    fov_range = np.linspace(20, 500, 40)

    # Background model
    def background_rate(sun_angle_deg, fov_urad):
        """Simplified PST-based background model"""
        sun_angle_rad = np.radians(sun_angle_deg)
        fov_rad = fov_urad * 1e-6

        log_pst = -4.0 - 3.0 * np.log10(sun_angle_deg / 10.0)
        log_pst = np.clip(log_pst, -10, -4)
        pst = 10 ** log_pst

        lambda_b = pst * (fov_rad / 100e-6) ** 2 * 1000
        return lambda_b

    # Create grids
    Sun_grid, FoV_grid = np.meshgrid(sun_angles, fov_range)
    Background_grid = np.zeros_like(Sun_grid)
    Capacity_grid = np.zeros_like(Sun_grid)

    # System parameters
    params = config['system_parameters']
    S_bar = params['Sbar']
    S_max = params['Smax']
    dt = params['dt']
    tau_d = params.get('tau_d')
    M_pixels = params['M_pixels']

    print("🔄 Computing background and capacity grids...")
    for i in tqdm(range(len(fov_range)), desc="Computing grid"):
        for j in range(len(sun_angles)):
            lambda_b = background_rate(Sun_grid[i, j], FoV_grid[i, j])
            Background_grid[i, j] = lambda_b

            C_lb, _ = capacity_lb(S_bar, S_max, lambda_b, dt, tau_d, M_pixels)
            Capacity_grid[i, j] = C_lb

    # ========================================================================
    # SINGLE CONTOUR LEVEL GENERATION (Linear spacing)
    # ========================================================================

        c_min, c_max = np.min(Capacity_grid), np.max(Capacity_grid)
        c_mean, c_median = np.mean(Capacity_grid), np.median(Capacity_grid)

        print(f"\n📊 Capacity Statistics:")
        print(f"   Range:  [{c_min:.4f}, {c_max:.4f}] bits/slot")
        print(f"   Mean:   {c_mean:.4f} bits/slot")
        print(f"   Median: {c_median:.4f} bits/slot")

        # Strategy: Minimalist approach with physical significance
        # - Left zone (high background): 1 contour only
        # - Right zone (main operation): 3 contours

        capacity_levels = np.array([
            0.90,  # High background (1 line in left zone)
            0.985,  # Transition zone
            0.993,  # Good operation
            0.997,  # Excellent conditions
            0.999  # Near-optimal
        ])

        # Verify levels are within data range
        capacity_levels = capacity_levels[(capacity_levels >= c_min) & (capacity_levels <= c_max)]
        capacity_levels = np.sort(capacity_levels)

        print(f"\n✅ Optimized contour levels (n={len(capacity_levels)}):")
        print(f"   Left zone:  [{capacity_levels[0]:.3f}]")
        print(f"   Right zone: {capacity_levels[1:]} bits/slot")
        print(f"\n   Physical interpretation:")
        print(f"   • 0.900: High background zone (avoid if possible)")
        print(f"   • 0.990: Minimum acceptable operation")
        print(f"   • 0.996: Typical good conditions")
        print(f"   • 0.999: Excellent link quality")
    # ========================================================================
    # Create figure
    # ========================================================================

    fig, ax = plt.subplots(figsize=(10, 7))

    # Background heatmap
    im = ax.pcolormesh(Sun_grid, FoV_grid, np.log10(Background_grid + 1e-10),
                       shading='auto', cmap='YlOrRd', alpha=0.8)

    # Capacity contours
    print(f"\n🎨 Drawing contours...")
    try:
        contours = ax.contour(Sun_grid, FoV_grid, Capacity_grid,
                              levels=capacity_levels,
                              colors='navy',  # 深蓝色
                              linewidths=2.5,
                              linestyles='solid')

        # # Label contours
        # labels = ax.clabel(contours, inline=True, fontsize=9,
        #                    fmt='%.3f', inline_spacing=10)
        #
        # # Add white background to labels for readability
        # for label in labels:
        #     label.set_bbox(dict(
        #         boxstyle='round,pad=0.3',
        #         facecolor='white',
        #         edgecolor='navy',
        #         alpha=0.8,
        #         linewidth=1.0
        #     ))

        print(f"✅ Successfully drew {len(capacity_levels)} contour lines")

        # Verify which contours were actually drawn
        print(f"   Contour levels in plot: {contours.levels}")

    except Exception as e:
        print(f"❌ Contour drawing FAILED: {e}")
        import traceback
        traceback.print_exc()

    # Colorbar for background
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(Background λb) [photons/slot]',
                   rotation=270, labelpad=20, fontsize=11, fontweight='bold')

    # Reference lines
    ax.axhline(y=50, color='blue', linestyle='--', linewidth=2, alpha=0.7,
               label='Typical FoV (50 μrad)')
    ax.axhline(y=200, color='green', linestyle='--', linewidth=2, alpha=0.7,
               label='Wide FoV (200 μrad)')
    ax.axvline(x=30, color='red', linestyle=':', linewidth=2.5, alpha=0.7,
               label='Min Sun Avoidance')

    # Labels and styling
    ax.set_xlabel('Sun Avoidance Angle [degrees]', fontweight='bold', fontsize=12)
    ax.set_ylabel('Receiver FoV [μrad]', fontweight='bold', fontsize=12)
    ax.set_title('Sun Avoidance Design Law\n(Background + Capacity Contours)',
                 fontweight='bold', fontsize=13)
    ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)

    plt.tight_layout()

    # Save figure
    output_path = f"{dirs['figures']}/Fig_4_Design_Law"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved: {output_path}.pdf")
    print(f"✅ Saved: {output_path}.png")

    # Save design data
    design_data = {
        'sun_angles': sun_angles,
        'fov_range': fov_range,
        'background_grid': Background_grid,
        'capacity_grid': Capacity_grid,
        'capacity_levels': capacity_levels
    }
    np.savez(f"{dirs['data']}/design_law_data.npz", **design_data)

    print(f"✅ Saved: design_law_data.npz")

# ============================================================================
# MONTE CARLO VALIDATION
# ============================================================================

# ============================================================================
# WORKER FUNCTION FOR MONTE CARLO (新增到文件开头导入之后)
# ============================================================================

def _monte_carlo_trial_worker(args):
    """
    Single Monte Carlo trial worker

    Args:
        args: (trial_id, max_pilots, scenario, params, dither_seq, S_pilot,
               mu_true, gamma, b, crlb_var_mux, crlb_var_muy)

    Returns:
        (mu_x_estimate, mu_y_estimate)
    """
    (trial_id, max_pilots, scenario, params, dither_seq, S_pilot,
     mu_true, gamma, b, crlb_var_mux, crlb_var_muy) = args

    # Set unique random seed for this trial
    np.random.seed(trial_id + 42)

    # Generate synthetic observations
    pilot_observations = []

    for n in range(max_pilots):
        # True pointing with dither
        mu_eff = mu_true + dither_seq[n % len(dither_seq)]

        # Expected pointing loss
        L_p = (1.0 / gamma) * np.exp(-b * np.dot(mu_eff, mu_eff) / gamma)

        # Signal photon rate
        lambda_signal = S_pilot * L_p
        lambda_total = lambda_signal + scenario['r_b'] * params['dt']

        # Generate Poisson observation
        y_obs = np.random.poisson(lambda_total)
        pilot_observations.append((y_obs, dither_seq[n % len(dither_seq)]))

    # MLE estimation (grid search)
    def log_likelihood(mu_est):
        ll = 0.0
        for y_obs, dither in pilot_observations:
            mu_eff = mu_est + dither
            L_p = (1.0 / gamma) * np.exp(-b * np.dot(mu_eff, mu_eff) / gamma)
            lambda_pred = S_pilot * L_p + scenario['r_b'] * params['dt']

            if lambda_pred > 0:
                ll += y_obs * np.log(lambda_pred) - lambda_pred

        return ll

    # Grid search MLE
    best_mu = mu_true.copy()
    best_ll = log_likelihood(mu_true)

    # Search in neighborhood
    search_range = 5 * np.sqrt(crlb_var_mux)
    mu_search = np.linspace(-search_range, search_range, 21)

    for mux_test in mu_search:
        for muy_test in mu_search:
            mu_test = np.array([mux_test, muy_test])
            ll = log_likelihood(mu_test)
            if ll > best_ll:
                best_ll = ll
                best_mu = mu_test

    return (best_mu[0], best_mu[1])


# ============================================================================
# MONTE CARLO VALIDATION (并行版本)
# ============================================================================

def monte_carlo_validation(config, dirs, n_workers=None):
    """
    Monte Carlo validation of CRLB bounds

    ** PARALLEL VERSION **

    Parameters:
    -----------
    n_workers : int or None
        Number of parallel workers for MC trials.
    """
    print("\n" + "=" * 60)
    print("🎲 MONTE CARLO CRLB VALIDATION (PARALLEL)")
    print("=" * 60)

    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    n_workers = max(1, n_workers)

    print(f"🚀 Using {n_workers} parallel worker(s) for MC trials")

    setup_ieee_style()

    # Parameters for validation
    params = config['system_parameters']
    n_trials = config['monte_carlo']['n_trials']

    # Test scenarios
    test_scenarios = [
        {'Sbar': 10, 'r_b': 0.1, 'alpha': 0.3, 'rho': 0.5, 'name': 'Low Background'},
        {'Sbar': 50, 'r_b': 1.0, 'alpha': 0.2, 'rho': 0.6, 'name': 'Medium Background'},
        {'Sbar': 20, 'r_b': 5.0, 'alpha': 0.4, 'rho': 0.7, 'name': 'High Background'}
    ]

    fig, axes = plt.subplots(1, len(test_scenarios), figsize=(15, 5))
    if len(test_scenarios) == 1:
        axes = [axes]

    for idx, scenario in enumerate(test_scenarios):
        print(f"\n🔬 Testing scenario: {scenario['name']}")

        # Setup parameters
        params_test = params.copy()
        params_test.update(scenario)

        # True parameter values
        mu_true = np.array([1.0e-6, 0.5e-6])

        # Generate dither sequence
        max_pilots = int(scenario['alpha'] * params['N'])
        dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

        # Compute theoretical CRLB
        I_pilot = fim_pilot(scenario['alpha'], scenario['rho'], scenario['Sbar'],
                            params['N'], params['dt'], params_test, dither_seq,
                            params.get('tau_d'), None, params['M_pixels'])

        J = I_pilot + params['J_P'] + 1e-12 * np.eye(4)
        if np.linalg.cond(J) < 1e30:
            crlb_cov = np.linalg.inv(J)
            crlb_var_mux = crlb_cov[0, 0]
            crlb_var_muy = crlb_cov[1, 1]
        else:
            print(f"  ⚠️ Singular FIM for {scenario['name']}")
            continue

        # Precompute constants
        a = 4.0 / (params['theta_b'] ** 2)
        b = 2.0 / (params['theta_b'] ** 2)
        gamma = 1.0 + a * params['sigma2']
        S_pilot = scenario['rho'] * scenario['Sbar'] / scenario['alpha']

        # Prepare arguments for parallel workers
        worker_args = [
            (trial, max_pilots, scenario, params, dither_seq, S_pilot,
             mu_true, gamma, b, crlb_var_mux, crlb_var_muy)
            for trial in range(n_trials)
        ]

        # Execute Monte Carlo trials in parallel
        print(f"  🔄 Running {n_trials} Monte Carlo trials in parallel...")

        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                results = list(tqdm(
                    pool.imap(_monte_carlo_trial_worker, worker_args),
                    total=n_trials,
                    desc=f"  MC trials ({scenario['name']})"
                ))
        else:
            results = [
                _monte_carlo_trial_worker(args)
                for args in tqdm(worker_args, desc=f"  MC trials ({scenario['name']})")
            ]

        # Extract estimates
        estimates_mux = [r[0] for r in results]
        estimates_muy = [r[1] for r in results]

        # Compute sample variances
        sample_var_mux = np.var(estimates_mux)
        sample_var_muy = np.var(estimates_muy)

        # Plot comparison
        ax = axes[idx]

        # CRLB bounds (horizontal lines)
        ax.axhline(y=crlb_var_mux, color='red', linewidth=3,
                   label=f'CRLB μx: {crlb_var_mux:.2e}')
        ax.axhline(y=crlb_var_muy, color='blue', linewidth=3,
                   label=f'CRLB μy: {crlb_var_muy:.2e}')

        # Sample variances (points with error bars)
        ax.errorbar([1], [sample_var_mux], yerr=sample_var_mux * 0.1,
                    fmt='ro', markersize=10, capsize=5,
                    label=f'Sample μx: {sample_var_mux:.2e}')
        ax.errorbar([2], [sample_var_muy], yerr=sample_var_muy * 0.1,
                    fmt='bo', markersize=10, capsize=5,
                    label=f'Sample μy: {sample_var_muy:.2e}')

        ax.set_yscale('log')
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['μx', 'μy'])
        ax.set_ylabel('Variance [rad²]')
        ax.set_title(f'{scenario["name"]}\n({n_trials} trials)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Print results
        efficiency_mux = crlb_var_mux / sample_var_mux * 100
        efficiency_muy = crlb_var_muy / sample_var_muy * 100
        print(f"  📊 Results for {scenario['name']}:")
        print(f"     μx efficiency: {efficiency_mux:.1f}%")
        print(f"     μy efficiency: {efficiency_muy:.1f}%")

    plt.tight_layout()

    # Save figure
    output_path = f"{dirs['appendix']}/Fig_Appendix_CRLB_Validation"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}.pdf")

# ============================================================================
# ERGODIC VS OUTAGE ANALYSIS
# ============================================================================

def ergodic_vs_outage_analysis(config, dirs):
    """
    Ergodic vs Outage capacity analysis for time-varying channels
    """
    print("\n" + "=" * 60)
    print("📡 ERGODIC vs OUTAGE ANALYSIS")
    print("=" * 60)

    setup_ieee_style()

    # Define background distribution (discrete)
    background_states = {
        'Low': {'lambda_b': 0.01, 'probability': 0.70},
        'Medium': {'lambda_b': 1.0, 'probability': 0.25},
        'High': {'lambda_b': 10.0, 'probability': 0.05}
    }

    # Signal budget range
    S_bar_range = np.logspace(-1, 2, 30)

    # System parameters
    params = config['system_parameters']
    outage_probability = 0.1  # 10% outage

    ergodic_rates = []
    outage_rates = []

    print("🔄 Computing Ergodic and Outage rates...")

    for S_bar in tqdm(S_bar_range):
        # Compute capacity for each background state
        capacities = []
        probabilities = []

        for state_name, state_info in background_states.items():
            lambda_b = state_info['lambda_b']
            prob = state_info['probability']

            C_lb, _ = capacity_lb(S_bar, params['Smax'], lambda_b,
                                  params['dt'], params.get('tau_d'), params['M_pixels'])

            capacities.append(C_lb)
            probabilities.append(prob)

        # Ergodic capacity: E[C(lambda_b)]
        ergodic_rate = sum(c * p for c, p in zip(capacities, probabilities))
        ergodic_rates.append(ergodic_rate)

        # Outage capacity: quantile of C(lambda_b)
        # For discrete distribution, find the quantile
        sorted_caps_probs = sorted(zip(capacities, probabilities))
        cumulative_prob = 0.0
        outage_rate = 0.0

        for cap, prob in sorted_caps_probs:
            cumulative_prob += prob
            if cumulative_prob >= outage_probability:
                outage_rate = cap
                break

        outage_rates.append(outage_rate)

    # Plot results
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.semilogx(S_bar_range, ergodic_rates, 'b-', linewidth=3,
                marker='o', markersize=5, label='Ergodic Rate')
    ax.semilogx(S_bar_range, outage_rates, 'r--', linewidth=3,
                marker='s', markersize=5, label=f'Outage Rate ({outage_probability * 100:.0f}%)')

    # Fill area between curves
    ax.fill_between(S_bar_range, outage_rates, ergodic_rates,
                    alpha=0.3, color='gray', label='Rate Loss Region')

    # Mark background regime transitions
    ax.axvline(x=1, color='orange', alpha=0.5, linestyle=':',
               label='Typical Transition')

    ax.set_xlabel('Average Signal Budget S̄ [photons/slot]', fontweight='bold')
    ax.set_ylabel('Rate [bits/slot]', fontweight='bold')
    ax.set_title('Ergodic vs Outage Capacity\n(Time-Varying Background Channel)', fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Add text annotation
    ax.text(0.5, 0.7,
            f'Background States:\n'
            f'• Low (λb=0.01): {background_states["Low"]["probability"] * 100:.0f}%\n'
            f'• Medium (λb=1.0): {background_states["Medium"]["probability"] * 100:.0f}%\n'
            f'• High (λb=10.0): {background_states["High"]["probability"] * 100:.0f}%',
            transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    # Save figure
    output_path = f"{dirs['appendix']}/Fig_Appendix_Ergodic_Outage"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}.pdf")

    # Save data
    ergodic_data = {
        'S_bar_range': S_bar_range,
        'ergodic_rates': ergodic_rates,
        'outage_rates': outage_rates,
        'background_states': background_states
    }
    np.savez(f"{dirs['data']}/ergodic_outage_data.npz", **ergodic_data)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Section IV: Numerical Results Generator')
    parser.add_argument('--figure', type=str, default='all',
                        help='Figure to generate: 1,2,3,4,all')
    parser.add_argument('--validation', action='store_true',
                        help='Run Monte Carlo validation')
    parser.add_argument('--ergodic', action='store_true',
                        help='Run Ergodic vs Outage analysis')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Configuration file path')
    parser.add_argument('--diagnose-fig4', action='store_true',
                        help='Run diagnostic for Figure 4')

    args = parser.parse_args()

    print("🚀 OISL-ISAC Section IV: Numerical Results Generator")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    set_reproducibility(args.seed)
    config = load_config(args.config)
    dirs = setup_directories()

    # Generate figures
    if args.figure == 'all' or args.figure == '1':
        generate_figure_1(config, dirs)

    if args.figure == 'all' or args.figure == '2':
        generate_figure_2(config, dirs)

    if args.figure == 'all' or args.figure == '3':
        generate_figure_3(config, dirs)

    # 在生成 Figure 4 之前，先运行诊断
    if args.diagnose_fig4:
        diag_results = diagnose_figure_4(config, dirs)
        print("\n🔍 Diagnostic complete. Review output above.")
        return


    if args.figure == 'all' or args.figure == '4':
        generate_figure_4(config, dirs)

    # Additional analyses
    if args.validation or args.figure == 'all':
        monte_carlo_validation(config, dirs)

    if args.ergodic or args.figure == 'all':
        ergodic_vs_outage_analysis(config, dirs)

    # Summary
    print(f"\n{'=' * 60}")
    print("✅ SECTION IV GENERATION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"📁 Results saved in: {dirs['base']}/")

    # List generated files
    for subdir in ['figures', 'appendix', 'data']:
        subdir_path = dirs[subdir]
        if os.path.exists(subdir_path):
            files = [f for f in os.listdir(subdir_path) if f.endswith(('.pdf', '.png', '.npz'))]
            if files:
                print(f"\n📊 {subdir.title()}:")
                for file in sorted(files):
                    print(f"   • {file}")

    print(f"\n🎯 To reproduce: python {sys.argv[0]} --seed {args.seed}")


if __name__ == "__main__":
    main()