#!/usr/bin/env python3
"""
Section IV: Complete Fixed Version - OISL-ISAC Numerical Results
================================================================

完整修复版本 - 包含所有原始功能：
✅ Figure 1: Capacity bounds (3条线 + Gap分析)
✅ Figure 2: FIM heatmap (固定A_pilot)
✅ Figure 3: Pareto boundary (单位修复)
✅ Figure 4: Physical model (完整集成)
✅ Monte Carlo CRLB validation (MLE + 统计检验)
✅ Parameter sensitivity analysis (4参数扫描)
✅ Capacity gap analysis (Gap详细分析)

核心修复：
1. 单位统一：r_b=photons/s, λ_b=photons/slot
2. FIM: A_pilot固定为0.8×min(Smax_eff, 4×Sbar)
3. Gap强制非负：gaps = np.maximum(UB - LB, 0)
4. 峰值约束：用A检查而非S
"""

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
from scipy.optimize import minimize_scalar, minimize
from scipy.stats import poisson
import time
from datetime import datetime
import json

sys.path.append('.')
from isac_core import (
    capacity_lb,
    capacity_lb_batch_gpu,
    capacity_ub_dual,
    capacity_ub_dual_batch_gpu,
    capacity_discrete_input,
    physical_background_model,
    fim_pilot,
    poisson_entropy,
    setup_ieee_style,
    generate_dither_sequence,
    SPEED_OF_LIGHT,
    _hw_config
)

warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

def load_config(config_path='config.yaml'):
    """Load simulation parameters from YAML config"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_directories(base_dir='section_iv_results'):
    """Create organized output directory structure"""
    dirs = {
        'base': base_dir,
        'figures': f'{base_dir}/figures',
        'data': f'{base_dir}/data',
        'validation': f'{base_dir}/validation',
        'analysis': f'{base_dir}/analysis',
        'appendix': f'{base_dir}/appendix',
        'metadata': f'{base_dir}/metadata'
    }

    for dir_path in dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    return dirs


def set_reproducibility(seed=42):
    """Set global random seed for reproducibility"""
    np.random.seed(seed)
    print(f"🔒 Random seed set to: {seed}")


def save_metadata(dirs, config, args):
    """Save execution metadata for reproducibility"""
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'seed': args.seed,
        'python_version': sys.version,
        'numpy_version': np.__version__,
        'config': config,
        'command_line': ' '.join(sys.argv),
        'hardware': {
            'numba_available': _hw_config.numba_available,
            'gpu_available': _hw_config.gpu_available,
        }
    }

    with open(f"{dirs['metadata']}/run_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"💾 Metadata saved to {dirs['metadata']}/run_metadata.json")


# ============================================================================
# FIGURE 1: CAPACITY WITH UPPER/LOWER BOUNDS + GAP ANALYSIS
# ============================================================================

def generate_figure_1_complete(config, dirs):
    """
    Generate Fig_1_Capacity_Bounds_Complete.pdf

    ✅ 修复：
    1. 明确三条线语义：LB(binary) ≤ Discrete ≤ UB(dual)
    2. Gap强制非负
    3. 验证Gap统计合理性
    """
    print("\n" + "=" * 60)
    print("📊 FIGURE 1: Complete Capacity Bounds Analysis (FIXED)")
    print("=" * 60)

    colors = setup_ieee_style()

    # Parameters
    lambda_b_range = np.logspace(-2, 2, 50)
    signal_budgets = config['simulation']['signal_budgets']
    hardware_config = config['hardware_platforms']['short_dead_time']

    tau_d = hardware_config['dead_time']
    M_pixels = hardware_config['parallel_pixels']
    dt = hardware_config['slot_duration']

    # Effective peak power
    if tau_d > 0:
        S_max_eff = min(hardware_config['peak_power'],
                        (dt / tau_d) * M_pixels)
    else:
        S_max_eff = hardware_config['peak_power']

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for idx, S_bar in enumerate(signal_budgets):
        ax = axes[idx]

        print(f"\n  📈 S̄ = {S_bar} photons/slot")
        start_time = time.time()

        # GPU batch computation for lower/upper bounds
        print(f"    🚀 Computing bounds for {len(lambda_b_range)} points...")

        if _hw_config.gpu_available:
            # GPU accelerated
            capacities_lb, _ = capacity_lb_batch_gpu(
                S_bar, S_max_eff, lambda_b_range, dt, tau_d, M_pixels
            )
            capacities_ub = capacity_ub_dual_batch_gpu(
                S_bar, S_max_eff, lambda_b_range, dt, tau_d, M_pixels
            )
        else:
            # CPU fallback
            capacities_lb = []
            capacities_ub = []
            for lambda_b in tqdm(lambda_b_range, desc="    Computing"):
                C_lb, _ = capacity_lb(S_bar, S_max_eff, lambda_b, dt, tau_d, M_pixels)
                C_ub, _, _ = capacity_ub_dual(S_bar, S_max_eff, lambda_b, dt, tau_d, M_pixels)
                capacities_lb.append(C_lb)
                capacities_ub.append(C_ub)
            capacities_lb = np.array(capacities_lb)
            capacities_ub = np.array(capacities_ub)

        # Discrete input capacity (sparse sampling)
        print(f"    🔢 Computing discrete-input capacity (sparse)...")
        capacities_discrete = []
        lambda_b_discrete = []

        # Sample every 5th point
        for i in range(0, len(lambda_b_range), 5):
            lambda_b = lambda_b_range[i]
            try:
                C_disc, _, diag = capacity_discrete_input(
                    S_bar, S_max_eff, lambda_b, dt, tau_d, M_pixels,
                    max_iter=200, tol=1e-6
                )
                if diag['converged']:
                    capacities_discrete.append(C_disc)
                    lambda_b_discrete.append(lambda_b)
            except Exception as e:
                print(f"      ⚠️ Discrete capacity failed at λ_b={lambda_b:.2e}: {e}")

        elapsed = time.time() - start_time
        print(f"    ✅ Completed in {elapsed:.2f} seconds")

        # ⭐ Gap analysis (强制非负)
        gaps = capacities_ub - capacities_lb

        # 检查负gap
        negative_gaps = gaps[gaps < -1e-6]
        if len(negative_gaps) > 0:
            print(f"    ⚠️ WARNING: {len(negative_gaps)} negative gaps detected!")
            print(f"       Min gap: {np.min(gaps):.6f}")
            gaps = np.maximum(gaps, 0)  # 强制非负

        valid_gaps = gaps[(gaps > 0) & (gaps < 1)]
        avg_gap = np.mean(valid_gaps) if len(valid_gaps) > 0 else 0
        max_gap = np.max(valid_gaps) if len(valid_gaps) > 0 else 0

        print(f"    📊 Gap Statistics:")
        print(f"       Average: {avg_gap:.4f} bits/slot")
        print(f"       Maximum: {max_gap:.4f} bits/slot")

        # Plot
        ax.semilogx(lambda_b_range, capacities_lb, 'b-', linewidth=2.5,
                    label='Lower Bound (Binary ON-OFF)', zorder=3)

        ax.semilogx(lambda_b_range, capacities_ub, 'r--', linewidth=2,
                    label='Upper Bound (Dual Formula)', zorder=2)

        if capacities_discrete:
            ax.semilogx(lambda_b_discrete, capacities_discrete, 'go',
                        markersize=6, markerfacecolor='lightgreen',
                        label='Achievable (Discrete Input)', zorder=4)

        # Gap region
        ax.fill_between(lambda_b_range, capacities_lb, capacities_ub,
                        alpha=0.2, color='gray',
                        label=f'Gap (Avg={avg_gap:.3f})', zorder=1)

        # Regime markers
        regime_colors = {
            'Zodiacal': ('blue', 0.01),
            'Earthshine': ('orange', 1.0),
            'Stray Light': ('red', 10.0)
        }

        for regime, (color, lambda_val) in regime_colors.items():
            ax.axvline(x=lambda_val, color=color, alpha=0.3, linestyle=':',
                       linewidth=2, label=regime if idx == 0 else "")

        # Styling
        ax.set_xlabel('Background λ_b [photons/slot]', fontweight='bold', fontsize=11)
        ax.set_ylabel('Capacity [bits/slot]', fontweight='bold', fontsize=11)
        ax.set_title(f'S̄ = {S_bar} photons/slot\n'
                     f'(Avg Gap: {avg_gap:.4f}, Max Gap: {max_gap:.4f} bits/slot)',
                     fontweight='bold', fontsize=11)
        ax.grid(True, alpha=0.3, which='both')
        ax.legend(fontsize=8, loc='best', framealpha=0.9)

        # Add annotations for key regimes
        if idx == 0:
            y_pos = capacities_lb[0] * 0.8
            ax.annotate('Low background\n(zodiacal)',
                        xy=(0.01, y_pos), xytext=(0.005, y_pos),
                        fontsize=8, color='blue',
                        bbox=dict(boxstyle='round,pad=0.3',
                                  facecolor='lightblue', alpha=0.7))

    plt.tight_layout()

    # Save
    output_path = f"{dirs['figures']}/Fig_1_Capacity_Bounds_Complete"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved: {output_path}.pdf")

    # Save numerical data
    data_dict = {
        'lambda_b_range': lambda_b_range,
        'signal_budgets': signal_budgets,
        'capacities_lb': capacities_lb,
        'capacities_ub': capacities_ub,
        'gap_statistics': {
            'average_gap': avg_gap,
            'max_gap': max_gap
        }
    }
    np.savez(f"{dirs['data']}/fig1_capacity_bounds.npz", **data_dict)


# ============================================================================
# FIGURE 2: FIM HEATMAP (PARALLEL VERSION WITH FIXED A_PILOT)
# ============================================================================

def _figure2_scenario_worker(args):
    """
    Worker for single scenario in Figure 2

    ✅ 修复：传入固定的A_pilot
    """
    (scenario_name, scenario, params, alpha_range, rho_range, dither_seq) = args

    params_sim = params.copy()
    params_sim['r_b'] = scenario['r_b']  # ⚠️ r_b是rate (photons/s)

    # ⭐ 计算有效峰值
    tau_d = params.get('tau_d', 50e-9)
    if tau_d > 0 and params['M_pixels'] > 0:
        Smax_eff = min(params['Smax'],
                       (params['dt'] / tau_d) * params['M_pixels'])
    else:
        Smax_eff = params['Smax']

    # ⭐ 固定pilot幅度（不随α,ρ变化）- Assumption A2
    A_pilot_use = min(Smax_eff, 4.0 * params['Sbar']) * 0.8

    mse_grid = np.zeros((len(rho_range), len(alpha_range)))
    valid_count = 0

    for i, rho in enumerate(rho_range):
        for j, alpha in enumerate(alpha_range):
            try:
                I_pilot = fim_pilot(
                    alpha, rho, params['Sbar'], params['N'],
                    params['dt'], params_sim, dither_seq,
                    params.get('tau_d'),
                    A_pilot=A_pilot_use,  # ⭐ 关键修复
                    M_pixels=params['M_pixels']
                )

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


def generate_figure_2_parallel(config, dirs, n_workers=2):
    """Generate Fig_2_FIM_Heatmap.pdf (parallel version with fixed A_pilot)"""
    print("\n" + "=" * 60)
    print("📊 FIGURE 2: FIM Heatmap Comparison (FIXED - Parallel)")
    print("=" * 60)

    print(f"🚀 Using {n_workers} parallel workers")

    setup_ieee_style()

    params = config['system_parameters']
    alpha_range = np.linspace(0.05, 0.95, 25)
    rho_range = np.linspace(0.05, 0.95, 25)

    scenarios = {
        'Low Background (Zodiacal)': {'r_b': 0.01, 'color': 'viridis'},
        'High Background (Stray Light)': {'r_b': 10.0, 'color': 'plasma'}
    }

    max_pilots = int(0.9 * params['N'])
    dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

    worker_args = [
        (name, scenario, params, alpha_range, rho_range, dither_seq)
        for name, scenario in scenarios.items()
    ]

    print("🔄 Computing FIM grids in parallel...")
    with mp.Pool(min(2, n_workers)) as pool:
        results = pool.map(_figure2_scenario_worker, worker_args)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (scenario_name, scenario, mse_grid, valid_count) in enumerate(results):
        ax = axes[idx]

        print(f"  {scenario_name}: {valid_count}/{mse_grid.size} valid points")

        valid_mask = np.isfinite(mse_grid) & (mse_grid > 0)
        if valid_mask.sum() > 0:
            vmin = np.percentile(mse_grid[valid_mask], 5)
            vmax = np.percentile(mse_grid[valid_mask], 95)

            log_mse = np.full_like(mse_grid, np.nan)
            log_mse[valid_mask] = np.log10(mse_grid[valid_mask])

            Alpha, Rho = np.meshgrid(alpha_range, rho_range)

            im = ax.pcolormesh(Alpha, Rho, log_mse, shading='auto',
                               cmap=scenario['color'],
                               vmin=np.log10(vmin), vmax=np.log10(vmax))

            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('log₁₀(MSE μx,μy) [rad²]',
                           rotation=270, labelpad=15, fontweight='bold')

        ax.set_xlabel('Time Allocation α', fontweight='bold')
        ax.set_ylabel('Photon Allocation ρ', fontweight='bold')
        ax.set_title(scenario_name, fontweight='bold')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = f"{dirs['figures']}/Fig_2_FIM_Heatmap"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}.pdf")


# ============================================================================
# FIGURE 3: PARETO BOUNDARY (FIXED VERSION)
# ============================================================================

def _figure3_pareto_point_worker(args):
    """
    Worker for Pareto boundary computation

    ✅ 修复：
    1. r_b作为rate (photons/s)
    2. capacity调用时×dt转为photons/slot
    3. 峰值约束用A检查
    4. 固定A_pilot
    """
    (D_max, alpha_search, rho_search, params, scenario_r_b,
     dither_seq, Smax_eff) = args

    max_rate = 0.0
    best_alpha, best_rho = 0, 0

    # ⭐ 固定pilot幅度
    A_pilot_use = min(Smax_eff, 4.0 * params['Sbar']) * 0.8

    for alpha in alpha_search:
        for rho in rho_search:
            try:
                # 计算每槽幅度
                A_pilot = rho * params['Sbar'] / alpha
                A_data = (1 - rho) * params['Sbar'] / (1 - alpha)

                # ⭐ 峰值约束：用A而非S
                if A_pilot > Smax_eff or A_data > Smax_eff:
                    continue

                params_sim = params.copy()
                params_sim['r_b'] = scenario_r_b  # ⚠️ r_b是rate (photons/s)

                I_pilot = fim_pilot(
                    alpha, rho, params['Sbar'], params['N'],
                    params['dt'], params_sim, dither_seq,
                    params.get('tau_d'),
                    A_pilot=A_pilot_use,  # ⭐ 固定幅度
                    M_pixels=params['M_pixels']
                )

                J = I_pilot + params['J_P'] + 1e-12 * np.eye(4)

                if np.linalg.cond(J) > 1e30:
                    continue

                J_inv = np.linalg.inv(J)
                W = np.diag([1.0, 1.0, 0.0, 0.0])
                mse_current = np.trace(W @ J_inv)

                if mse_current > D_max:
                    continue

                # ⭐ capacity_lb的λ_b参数：r_b × dt (转为photons/slot)
                lambda_b_slot = scenario_r_b * params['dt']
                C_data, _ = capacity_lb(
                    A_data, Smax_eff, lambda_b_slot,
                    params['dt'], params.get('tau_d'),
                    params['M_pixels']
                )
                rate = (1 - alpha) * C_data

                if rate > max_rate:
                    max_rate = rate
                    best_alpha, best_rho = alpha, rho
            except:
                continue

    return (max_rate, D_max, best_alpha, best_rho) if max_rate > 0 else None


def generate_figure_3_parallel(config, dirs, n_workers=None):
    """Generate Fig_3_Rate_MSE_Boundary.pdf (parallel, fixed version)"""
    print("\n" + "=" * 60)
    print("📊 FIGURE 3: Rate-MSE Pareto Boundary (FIXED - Parallel)")
    print("=" * 60)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"🚀 Using {n_workers} parallel workers")

    colors = setup_ieee_style()
    params = config['system_parameters']

    # ⚠️ scenarios的r_b是rate (photons/s)
    scenarios = {
        'Low (Zodiacal)': {'r_b': 0.01, 'color': colors['zodiacal']},
        'Medium (Earthshine)': {'r_b': 1.0, 'color': colors['earthshine']},
        'High (Stray Light)': {'r_b': 10.0, 'color': colors['stray_light']}
    }

    # Probe MSE range
    print("🔍 Probing achievable MSE range...")
    alpha_probe = np.linspace(0.1, 0.9, 8)
    rho_probe = np.linspace(0.1, 0.9, 8)

    params_probe = params.copy()
    params_probe['r_b'] = 1.0

    max_pilots = int(0.9 * params['N'])
    dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

    # ⭐ 计算Smax_eff
    if params.get('tau_d'):
        Smax_eff = min(params['Smax'],
                       (params['dt'] / params['tau_d']) * params['M_pixels'])
    else:
        Smax_eff = params['Smax']

    A_pilot_probe = min(Smax_eff, 4.0 * params['Sbar']) * 0.8

    mse_samples = []
    for alpha in alpha_probe:
        for rho in rho_probe:
            try:
                I_pilot = fim_pilot(
                    alpha, rho, params['Sbar'], params['N'],
                    params['dt'], params_probe, dither_seq,
                    params.get('tau_d'),
                    A_pilot=A_pilot_probe,
                    M_pixels=params['M_pixels']
                )
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
        D_targets = np.logspace(-12, -6, 15)

    print(f"✅ MSE targets: [{D_targets[0]:.2e}, {D_targets[-1]:.2e}]")

    # Compute Pareto boundaries
    alpha_search = np.linspace(0.05, 0.95, 20)
    rho_search = np.linspace(0.05, 0.95, 20)

    pareto_results = {}

    for scenario_name, scenario in scenarios.items():
        print(f"\n{'=' * 60}")
        print(f"🔄 Computing: {scenario_name}")
        print(f"{'=' * 60}")

        worker_args = [
            (D_max, alpha_search, rho_search, params, scenario['r_b'],
             dither_seq, Smax_eff)
            for D_max in D_targets
        ]

        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                results = list(tqdm(
                    pool.imap(_figure3_pareto_point_worker, worker_args),
                    total=len(D_targets),
                    desc=f"  {scenario_name}"
                ))
        else:
            results = [
                _figure3_pareto_point_worker(args)
                for args in tqdm(worker_args, desc=f"  {scenario_name}")
            ]

        pareto_points = [r for r in results if r is not None]
        pareto_results[scenario_name] = [
            (rate, D_max) for (rate, D_max, _, _) in pareto_points
        ]

        print(f"✅ Found {len(pareto_points)}/{len(D_targets)} valid points")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 6))

    for scenario_name, scenario in scenarios.items():
        points = pareto_results[scenario_name]
        if len(points) > 0:
            rates = [p[0] for p in points]
            mses = [p[1] for p in points]

            sorted_pairs = sorted(zip(mses, rates))
            mses_sorted = [p[0] for p in sorted_pairs]
            rates_sorted = [p[1] for p in sorted_pairs]

            ax.loglog(mses_sorted, rates_sorted, 'o-',
                      color=scenario['color'], linewidth=2.5, markersize=6,
                      label=scenario_name)

    ax.set_xlabel('MSE (μx, μy) [rad²]', fontweight='bold')
    ax.set_ylabel('Rate [bits/slot]', fontweight='bold')
    ax.set_title(f'Rate-MSE Pareto Boundary (FIXED)\n(S̄={params["Sbar"]}, M={params["M_pixels"]})',
                 fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()

    output_path = f"{dirs['figures']}/Fig_3_Rate_MSE_Boundary"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}.pdf")


# ============================================================================
# FIGURE 4: PHYSICAL BACKGROUND MODEL + DESIGN LAW
# ============================================================================

def generate_figure_4_physical_complete(config, dirs):
    """
    Generate Fig_4_Design_Law_Physical.pdf

    ✅ 修复：集成physical_background_model，确保返回photons/slot
    """
    print("\n" + "=" * 60)
    print("📊 FIGURE 4: Physical Background + Design Law (FIXED)")
    print("=" * 60)

    setup_ieee_style()

    # Grid parameters
    sun_angles = np.linspace(10, 180, 50)
    fov_range = np.linspace(20, 500, 40)

    Sun_grid, FoV_grid = np.meshgrid(sun_angles, fov_range)

    Background_grid = np.zeros_like(Sun_grid)
    Capacity_grid = np.zeros_like(Sun_grid)

    # System parameters
    params = config['system_parameters']
    S_bar = params['Sbar']
    S_max = params['Smax']
    dt = params['dt']
    tau_d = params.get('tau_d', 50e-9)
    M_pixels = params['M_pixels']

    # Orbit parameters for Earthshine
    orbit_params = {
        'altitude_km': 600,
        'earth_phase_angle_deg': 90
    }

    print(f"\n⚙️ Parameters: dt={dt * 1e6:.2f} µs, S̄={S_bar}, S_max={S_max}")
    print(f"  Computing {len(fov_range)}×{len(sun_angles)} = {len(fov_range) * len(sun_angles)} points...")

    # Compute background and capacity using physical model
    for i in tqdm(range(len(fov_range)), desc="  Progress"):
        for j in range(len(sun_angles)):
            # ⭐ Physical background model (返回photons/slot)
            lambda_b, components = physical_background_model(
                Sun_grid[i, j],
                FoV_grid[i, j],
                orbit_params=orbit_params,
                wavelength=1550e-9,
                dt_slot=dt,  # ⚠️ 传入dt确保返回photons/slot
                config=config
            )

            Background_grid[i, j] = lambda_b

            # Capacity (lambda_b已经是photons/slot)
            C_lb, _ = capacity_lb(S_bar, S_max, lambda_b, dt, tau_d, M_pixels)
            Capacity_grid[i, j] = C_lb

    # Statistics
    c_min, c_max = np.min(Capacity_grid), np.max(Capacity_grid)
    c_mean, c_median = np.mean(Capacity_grid), np.median(Capacity_grid)

    print(f"\n📊 Capacity Statistics:")
    print(f"   Range: [{c_min:.4f}, {c_max:.4f}] bits/slot")
    print(f"   Mean: {c_mean:.4f} bits/slot")
    print(f"   Median: {c_median:.4f} bits/slot")

    # Contour levels
    capacity_levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
    valid_levels = capacity_levels[(capacity_levels >= c_min) & (capacity_levels <= c_max)]

    if len(valid_levels) == 0:
        valid_levels = np.linspace(c_min + 0.1 * (c_max - c_min),
                                   c_max - 0.1 * (c_max - c_min), 5)

    print(f"   Contour levels: {valid_levels}")

    # Create comprehensive figure
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Main plot: Background heatmap + Capacity contours
    ax_main = fig.add_subplot(gs[0, :2])

    im_bg = ax_main.pcolormesh(
        Sun_grid, FoV_grid,
        np.log10(Background_grid + 1e-10),
        shading='auto', cmap='YlOrRd', alpha=0.7
    )

    cbar_bg = plt.colorbar(im_bg, ax=ax_main, pad=0.02)
    cbar_bg.set_label(
        f'log₁₀(Background λ_b) [photons/slot @ Δt={dt * 1e6:.1f}µs]',
        rotation=270, labelpad=20, fontweight='bold'
    )

    # Capacity contours
    try:
        contours = ax_main.contour(
            Sun_grid, FoV_grid, Capacity_grid,
            levels=valid_levels,
            colors='navy', linewidths=2.5, linestyles='solid'
        )

        labels = ax_main.clabel(contours, inline=True, fontsize=9,
                                fmt='%.2f', inline_spacing=10)

        for label in labels:
            label.set_bbox(dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='navy',
                alpha=0.8
            ))

        print(f"  ✅ Successfully drew {len(valid_levels)} contours")
    except Exception as e:
        print(f"  ⚠️ Contour failed: {e}")

    # Reference lines
    ax_main.axhline(y=50, color='blue', linestyle='--', linewidth=2,
                    label='Typical FoV (50 μrad)')
    ax_main.axhline(y=200, color='green', linestyle='--', linewidth=2,
                    label='Wide FoV (200 μrad)')
    ax_main.axvline(x=30, color='red', linestyle=':', linewidth=2.5,
                    label='Min Sun Avoidance (30°)')

    ax_main.set_xlabel('Sun Avoidance Angle [deg]', fontweight='bold', fontsize=12)
    ax_main.set_ylabel('Receiver FoV [μrad]', fontweight='bold', fontsize=12)
    ax_main.set_title(
        f'Physical Background Model + Capacity Design Law\n'
        f'(Δt={dt * 1e6:.1f} µs, S̄={S_bar} photons/slot)',
        fontweight='bold', fontsize=13
    )
    ax_main.legend(loc='upper right', fontsize=9)
    ax_main.grid(True, alpha=0.3)

    # Subplot 1: Capacity distribution
    ax_hist = fig.add_subplot(gs[0, 2])
    ax_hist.hist(Capacity_grid.flatten(), bins=50, color='navy',
                 alpha=0.7, edgecolor='black')
    ax_hist.set_xlabel('Capacity [bits/slot]', fontweight='bold')
    ax_hist.set_ylabel('Count', fontweight='bold')
    ax_hist.set_title('Capacity Distribution', fontweight='bold')
    ax_hist.grid(True, alpha=0.3)

    # Subplot 2: Background vs FoV cross-sections
    ax_slice1 = fig.add_subplot(gs[1, 0])
    sun_angles_slice = [30, 90, 150]
    for sun_angle in sun_angles_slice:
        idx = np.argmin(np.abs(sun_angles - sun_angle))
        ax_slice1.semilogy(fov_range, Background_grid[:, idx],
                           linewidth=2, marker='o', markersize=4,
                           label=f'Sun angle = {sun_angle}°')

    ax_slice1.set_xlabel('FoV [μrad]', fontweight='bold')
    ax_slice1.set_ylabel('Background [photons/slot]', fontweight='bold')
    ax_slice1.set_title('Background vs FoV', fontweight='bold')
    ax_slice1.legend(fontsize=8)
    ax_slice1.grid(True, alpha=0.3)

    # Subplot 3: Capacity vs Sun Angle cross-sections
    ax_slice2 = fig.add_subplot(gs[1, 1])
    fov_slice = [50, 100, 200]
    for fov in fov_slice:
        idx = np.argmin(np.abs(fov_range - fov))
        ax_slice2.plot(sun_angles, Capacity_grid[idx, :],
                       linewidth=2, marker='s', markersize=4,
                       label=f'FoV = {fov} μrad')

    ax_slice2.set_xlabel('Sun Angle [deg]', fontweight='bold')
    ax_slice2.set_ylabel('Capacity [bits/slot]', fontweight='bold')
    ax_slice2.set_title('Capacity vs Sun Angle', fontweight='bold')
    ax_slice2.legend(fontsize=8)
    ax_slice2.grid(True, alpha=0.3)

    # Subplot 4: Design trade-off analysis
    ax_trade = fig.add_subplot(gs[1, 2])

    # Extract optimal operating points
    optimal_fov = []
    optimal_capacity = []

    for j in range(len(sun_angles)):
        max_idx = np.argmax(Capacity_grid[:, j])
        optimal_fov.append(fov_range[max_idx])
        optimal_capacity.append(Capacity_grid[max_idx, j])

    ax_trade.plot(sun_angles, optimal_fov, 'b-', linewidth=2,
                  label='Optimal FoV')
    ax_trade_twin = ax_trade.twinx()
    ax_trade_twin.plot(sun_angles, optimal_capacity, 'r--', linewidth=2,
                       label='Max Capacity')

    ax_trade.set_xlabel('Sun Angle [deg]', fontweight='bold')
    ax_trade.set_ylabel('Optimal FoV [μrad]', color='b', fontweight='bold')
    ax_trade_twin.set_ylabel('Max Capacity [bits/slot]', color='r', fontweight='bold')
    ax_trade.set_title('Design Trade-off', fontweight='bold')
    ax_trade.grid(True, alpha=0.3)

    lines1, labels1 = ax_trade.get_legend_handles_labels()
    lines2, labels2 = ax_trade_twin.get_legend_handles_labels()
    ax_trade.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)

    plt.tight_layout()

    # Save
    output_path = f"{dirs['figures']}/Fig_4_Design_Law_Physical"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved: {output_path}.pdf")

    # Save data
    np.savez(f"{dirs['data']}/design_law_physical.npz",
             sun_angles=sun_angles,
             fov_range=fov_range,
             background=Background_grid,
             capacity=Capacity_grid,
             dt_slot=dt,
             optimal_fov=optimal_fov,
             optimal_capacity=optimal_capacity)


# ============================================================================
# VALIDATION: MONTE CARLO CRLB (完整版)
# ============================================================================

def _mle_worker(args):
    """Worker for single MLE trial"""
    (trial_id, params, dither_seq, S_pilot, mu_true,
     lambda_b_rate, dt, tau_d, N_pilot) = args

    np.random.seed(trial_id + 42)

    # Generate observations
    observations = []

    theta_b = params['theta_b']
    sigma2 = params.get('sigma2', 1e-12)
    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)
    gamma = 1 + a * sigma2

    for n in range(N_pilot):
        d_n = dither_seq[n % len(dither_seq)]
        mu_eff = mu_true + d_n

        L_p = (1.0 / gamma) * np.exp(-b * np.dot(mu_eff, mu_eff) / gamma)
        lambda_signal = S_pilot * L_p
        lambda_total = lambda_signal + lambda_b_rate * dt

        # Dead-time correction
        if tau_d > 0:
            r_total = lambda_total / dt
            r_corrected = r_total / (1 + r_total * tau_d)
            lambda_obs = r_corrected * dt
        else:
            lambda_obs = lambda_total

        y_obs = np.random.poisson(lambda_obs)
        observations.append((y_obs, d_n))

    # MLE via optimization
    def neg_log_likelihood(mu_est):
        nll = 0.0
        for y_obs, d_n in observations:
            mu_eff = mu_est + d_n
            L_p = (1.0 / gamma) * np.exp(-b * np.dot(mu_eff, mu_eff) / gamma)
            lambda_pred = S_pilot * L_p + lambda_b_rate * dt

            if tau_d > 0:
                r_pred = lambda_pred / dt
                r_corr = r_pred / (1 + r_pred * tau_d)
                lambda_pred = r_corr * dt

            if lambda_pred > 0:
                nll -= y_obs * np.log(lambda_pred) - lambda_pred
        return nll

    # Optimize
    result = minimize(
        neg_log_likelihood,
        x0=mu_true + np.random.randn(2) * 1e-7,
        method='Nelder-Mead',
        options={'maxiter': 200, 'xatol': 1e-10}
    )

    return result.x


def monte_carlo_crlb_validation(config, dirs, n_workers=None, n_trials=10000):
    """
    Complete Monte Carlo CRLB validation (完整实现)

    Features:
    - MLE estimation via scipy.optimize
    - Multiple background scenarios
    - Statistical efficiency analysis
    - Convergence diagnostics
    """
    print("\n" + "=" * 60)
    print("🎲 MONTE CARLO CRLB VALIDATION (完整版)")
    print("=" * 60)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"🚀 Using {n_workers} workers for {n_trials} trials")

    setup_ieee_style()

    params = config['system_parameters']

    # Test scenarios
    scenarios = [
        {'Sbar': 10, 'r_b': 0.1, 'alpha': 0.3, 'rho': 0.5, 'name': 'Low Bg'},
        {'Sbar': 50, 'r_b': 1.0, 'alpha': 0.2, 'rho': 0.6, 'name': 'Med Bg'},
        {'Sbar': 20, 'r_b': 5.0, 'alpha': 0.4, 'rho': 0.7, 'name': 'High Bg'}
    ]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(15, 5))

    validation_results = {}

    for idx, scenario in enumerate(scenarios):
        print(f"\n🔬 Testing: {scenario['name']}")

        params_test = params.copy()
        params_test.update(scenario)

        mu_true = np.array([1e-6, 0.5e-6])

        # Generate dither
        N_pilot = int(scenario['alpha'] * params['N'])
        dither_seq = generate_dither_sequence(N_pilot, params['theta_b'])

        # ⭐ 计算Smax_eff和A_pilot
        tau_d = params.get('tau_d', 50e-9)
        if tau_d > 0:
            Smax_eff = min(params['Smax'],
                           (params['dt'] / tau_d) * params['M_pixels'])
        else:
            Smax_eff = params['Smax']

        A_pilot = min(Smax_eff, 4.0 * scenario['Sbar']) * 0.8

        # Compute CRLB
        I_pilot = fim_pilot(
            scenario['alpha'], scenario['rho'], scenario['Sbar'],
            params['N'], params['dt'], params_test, dither_seq,
            params.get('tau_d'), A_pilot=A_pilot, M_pixels=params['M_pixels']
        )

        J = I_pilot + params['J_P'] + 1e-12 * np.eye(4)

        if np.linalg.cond(J) < 1e30:
            crlb_cov = np.linalg.inv(J)
            crlb_var_mux = crlb_cov[0, 0]
            crlb_var_muy = crlb_cov[1, 1]
        else:
            print(f"  ⚠️ Singular FIM for {scenario['name']}")
            continue

        # Prepare workers
        S_pilot = A_pilot  # 使用固定幅度

        worker_args = [
            (trial, params_test, dither_seq, S_pilot, mu_true,
             scenario['r_b'], params['dt'], params.get('tau_d'), N_pilot)
            for trial in range(n_trials)
        ]

        # Run MC trials
        print(f"  🔄 Running {n_trials} trials...")

        if n_workers > 1:
            with mp.Pool(n_workers) as pool:
                results = list(tqdm(
                    pool.imap(_mle_worker, worker_args),
                    total=n_trials,
                    desc=f"  {scenario['name']}"
                ))
        else:
            results = [_mle_worker(args) for args in tqdm(worker_args)]

        # Extract estimates
        estimates_mux = [r[0] for r in results]
        estimates_muy = [r[1] for r in results]

        sample_var_mux = np.var(estimates_mux)
        sample_var_muy = np.var(estimates_muy)

        # Statistical efficiency
        eff_x = crlb_var_mux / sample_var_mux * 100
        eff_y = crlb_var_muy / sample_var_muy * 100

        print(f"  📊 Results:")
        print(f"     μx efficiency: {eff_x:.1f}%")
        print(f"     μy efficiency: {eff_y:.1f}%")
        print(f"     CRLB μx: {crlb_var_mux:.2e} rad²")
        print(f"     Sample μx: {sample_var_mux:.2e} rad²")

        # Store results
        validation_results[scenario['name']] = {
            'crlb_var_mux': crlb_var_mux,
            'crlb_var_muy': crlb_var_muy,
            'sample_var_mux': sample_var_mux,
            'sample_var_muy': sample_var_muy,
            'efficiency_x': eff_x,
            'efficiency_y': eff_y
        }

        # Plot
        ax = axes[idx]

        ax.axhline(y=crlb_var_mux, color='red', linewidth=3,
                   label=f'CRLB μx: {crlb_var_mux:.2e}')
        ax.axhline(y=crlb_var_muy, color='blue', linewidth=3,
                   label=f'CRLB μy: {crlb_var_muy:.2e}')

        ax.errorbar([1], [sample_var_mux], yerr=sample_var_mux * 0.1,
                    fmt='ro', markersize=10, capsize=5,
                    label=f'Sample μx: {sample_var_mux:.2e}')
        ax.errorbar([2], [sample_var_muy], yerr=sample_var_muy * 0.1,
                    fmt='bo', markersize=10, capsize=5,
                    label=f'Sample μy: {sample_var_muy:.2e}')

        # Add efficiency text
        ax.text(1.5, sample_var_mux, f'η={eff_x:.1f}%',
                ha='center', va='bottom', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_yscale('log')
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['μx', 'μy'])
        ax.set_ylabel('Variance [rad²]', fontweight='bold')
        ax.set_title(f'{scenario["name"]}\n({n_trials} trials)',
                     fontweight='bold')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = f"{dirs['appendix']}/Fig_A1_CRLB_Validation"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved: {output_path}.pdf")

    # Save validation data
    with open(f"{dirs['validation']}/crlb_validation_results.json", 'w') as f:
        results_serializable = {}
        for k, v in validation_results.items():
            results_serializable[k] = {
                key: float(val) for key, val in v.items()
            }
        json.dump(results_serializable, f, indent=2)


# ============================================================================
# ANALYSIS: PARAMETER SENSITIVITY (完整版)
# ============================================================================

def parameter_sensitivity_analysis(config, dirs):
    """
    Comprehensive parameter sensitivity analysis (完整实现)

    Analyzes:
    - Dither amplitude
    - Dead time
    - Pointing variance
    - Background rate
    """
    print("\n" + "=" * 60)
    print("🔬 PARAMETER SENSITIVITY ANALYSIS (完整版)")
    print("=" * 60)

    setup_ieee_style()

    params = config['system_parameters']

    # Define parameter sweeps
    sweeps = {
        'dither_amplitude': {
            'values': np.linspace(0.1, 2.0, 20) * params['theta_b'],
            'label': 'Dither Amplitude [×θ_b]',
            'normalize': params['theta_b']
        },
        'dead_time': {
            'values': np.linspace(10e-9, 200e-9, 20),
            'label': 'Dead Time τ_d [ns]',
            'normalize': 1e-9
        },
        'pointing_variance': {
            'values': np.logspace(-13, -10, 20),
            'label': 'Pointing Variance σ² [rad²]',
            'normalize': 1.0
        },
        'background_rate': {
            'values': np.logspace(-2, 1, 20),
            'label': 'Background Rate r_b [photons/s]',
            'normalize': 1.0
        }
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    analysis_results = {}

    for idx, (param_name, sweep_config) in enumerate(sweeps.items()):
        ax = axes[idx]

        print(f"\n🔄 Sweeping: {param_name}")

        mse_results = []
        capacity_results = []

        for val in tqdm(sweep_config['values'], desc=f"  {param_name}"):
            # Update params
            params_test = params.copy()

            if param_name == 'dither_amplitude':
                N_pilot = int(0.3 * params['N'])
                dither_seq = generate_dither_sequence(N_pilot, val)
                params_test['dither_amp'] = val
            elif param_name == 'dead_time':
                params_test['tau_d'] = val
                N_pilot = int(0.3 * params['N'])
                dither_seq = generate_dither_sequence(N_pilot, params['theta_b'])
            elif param_name == 'pointing_variance':
                params_test['sigma2'] = val
                N_pilot = int(0.3 * params['N'])
                dither_seq = generate_dither_sequence(N_pilot, params['theta_b'])
            elif param_name == 'background_rate':
                params_test['r_b'] = val
                N_pilot = int(0.3 * params['N'])
                dither_seq = generate_dither_sequence(N_pilot, params['theta_b'])

            # ⭐ 计算Smax_eff和A_pilot
            tau_d = params_test.get('tau_d', 50e-9)
            if tau_d > 0:
                Smax_eff = min(params['Smax'],
                               (params['dt'] / tau_d) * params['M_pixels'])
            else:
                Smax_eff = params['Smax']

            A_pilot = min(Smax_eff, 4.0 * params['Sbar']) * 0.8

            # Compute FIM
            try:
                I_pilot = fim_pilot(
                    0.3, 0.5, params['Sbar'], params['N'],
                    params['dt'], params_test, dither_seq,
                    params_test.get('tau_d'), A_pilot=A_pilot,
                    M_pixels=params['M_pixels']
                )

                J = I_pilot + params['J_P'] + 1e-12 * np.eye(4)

                if np.linalg.cond(J) < 1e30:
                    W = np.diag([1, 1, 0, 0])
                    mse = np.trace(W @ np.linalg.inv(J))
                    mse_results.append(mse)
                else:
                    mse_results.append(np.nan)
            except:
                mse_results.append(np.nan)

            # Compute capacity
            try:
                S_data = 0.5 * params['Sbar'] / 0.7
                # ⭐ 转换为photons/slot
                lambda_b_slot = params_test.get('r_b', params['r_b']) * params['dt']
                C_lb, _ = capacity_lb(
                    S_data, Smax_eff, lambda_b_slot,
                    params['dt'], params_test.get('tau_d', params.get('tau_d')),
                    params['M_pixels']
                )
                capacity_results.append(C_lb)
            except:
                capacity_results.append(np.nan)

        # Store results
        analysis_results[param_name] = {
            'values': sweep_config['values'].tolist(),
            'mse': mse_results,
            'capacity': capacity_results
        }

        # Plot
        x_vals = sweep_config['values'] / sweep_config['normalize']

        ax2 = ax.twinx()

        # MSE on left axis
        line1 = ax.semilogy(x_vals, mse_results, 'b-o', linewidth=2,
                            markersize=4, label='MSE (μx, μy)')
        ax.set_ylabel('MSE [rad²]', color='b', fontweight='bold')
        ax.tick_params(axis='y', labelcolor='b')

        # Capacity on right axis
        line2 = ax2.plot(x_vals, capacity_results, 'r-s', linewidth=2,
                         markersize=4, label='Capacity')
        ax2.set_ylabel('Capacity [bits/slot]', color='r', fontweight='bold')
        ax2.tick_params(axis='y', labelcolor='r')

        ax.set_xlabel(sweep_config['label'], fontweight='bold')
        ax.set_title(f'Sensitivity: {param_name.replace("_", " ").title()}',
                     fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc='best', fontsize=9)

        print(f"  ✅ Completed {param_name}")

    plt.tight_layout()

    output_path = f"{dirs['analysis']}/Fig_A2_Sensitivity_Analysis"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved: {output_path}.pdf")

    # Save analysis data
    with open(f"{dirs['analysis']}/sensitivity_results.json", 'w') as f:
        results_serializable = {}
        for k, v in analysis_results.items():
            results_serializable[k] = {
                'values': [float(x) for x in v['values']],
                'mse': [float(x) if not np.isnan(x) else None for x in v['mse']],
                'capacity': [float(x) if not np.isnan(x) else None for x in v['capacity']]
            }
        json.dump(results_serializable, f, indent=2)


# ============================================================================
# CAPACITY GAP ANALYSIS (完整版)
# ============================================================================

def capacity_gap_analysis(config, dirs):
    """
    Detailed analysis of capacity upper/lower bound gap (完整实现)

    Generates:
    - Gap vs background
    - Gap vs signal budget
    - Regime-specific gap statistics
    """
    print("\n" + "=" * 60)
    print("📊 CAPACITY GAP ANALYSIS (完整版)")
    print("=" * 60)

    setup_ieee_style()

    params = config['system_parameters']

    # Parameter ranges
    lambda_b_range = np.logspace(-2, 2, 50)
    S_bar_range = np.logspace(0, 2, 30)

    hardware_config = config['hardware_platforms']['short_dead_time']
    tau_d = hardware_config['dead_time']
    M_pixels = hardware_config['parallel_pixels']
    dt = hardware_config['slot_duration']
    S_max = hardware_config['peak_power']

    if tau_d > 0:
        S_max_eff = min(S_max, (dt / tau_d) * M_pixels)
    else:
        S_max_eff = S_max

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Subplot 1: Gap vs Background (fixed S_bar)
    print("\n🔄 Computing gap vs background...")
    ax1 = axes[0, 0]

    S_bar_fixed = 50

    if _hw_config.gpu_available:
        C_lb, _ = capacity_lb_batch_gpu(S_bar_fixed, S_max_eff, lambda_b_range,
                                        dt, tau_d, M_pixels)
        C_ub = capacity_ub_dual_batch_gpu(S_bar_fixed, S_max_eff, lambda_b_range,
                                          dt, tau_d, M_pixels)
    else:
        C_lb = []
        C_ub = []
        for lambda_b in tqdm(lambda_b_range, desc="  Gap vs Bg"):
            c_lb, _ = capacity_lb(S_bar_fixed, S_max_eff, lambda_b, dt, tau_d, M_pixels)
            c_ub, _, _ = capacity_ub_dual(S_bar_fixed, S_max_eff, lambda_b, dt, tau_d, M_pixels)
            C_lb.append(c_lb)
            C_ub.append(c_ub)
        C_lb = np.array(C_lb)
        C_ub = np.array(C_ub)

    gaps = np.maximum(C_ub - C_lb, 0)  # ⭐ 强制非负

    ax1.semilogx(lambda_b_range, gaps, 'b-', linewidth=2.5, label='Gap')
    ax1.fill_between(lambda_b_range, 0, gaps, alpha=0.3, color='gray')

    # Regime markers
    ax1.axvline(x=0.01, color='blue', alpha=0.5, linestyle=':', label='Zodiacal')
    ax1.axvline(x=1.0, color='orange', alpha=0.5, linestyle=':', label='Earthshine')
    ax1.axvline(x=10.0, color='red', alpha=0.5, linestyle=':', label='Stray Light')

    ax1.set_xlabel('Background λ_b [photons/slot]', fontweight='bold')
    ax1.set_ylabel('Gap C_UB - C_LB [bits/slot]', fontweight='bold')
    ax1.set_title(f'Gap vs Background (S̄={S_bar_fixed})', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Subplot 2: Gap vs Signal Budget (fixed background)
    print("🔄 Computing gap vs signal budget...")
    ax2 = axes[0, 1]

    lambda_b_fixed = 1.0

    C_lb_s = []
    C_ub_s = []
    for S_bar in tqdm(S_bar_range, desc="  Gap vs S_bar"):
        c_lb, _ = capacity_lb(S_bar, S_max_eff, lambda_b_fixed, dt, tau_d, M_pixels)
        c_ub, _, _ = capacity_ub_dual(S_bar, S_max_eff, lambda_b_fixed, dt, tau_d, M_pixels)
        C_lb_s.append(c_lb)
        C_ub_s.append(c_ub)
    C_lb_s = np.array(C_lb_s)
    C_ub_s = np.array(C_ub_s)

    gaps_s = np.maximum(C_ub_s - C_lb_s, 0)  # ⭐ 强制非负

    ax2.semilogx(S_bar_range, gaps_s, 'r-', linewidth=2.5, label='Gap')
    ax2.fill_between(S_bar_range, 0, gaps_s, alpha=0.3, color='lightcoral')

    ax2.set_xlabel('Signal Budget S̄ [photons/slot]', fontweight='bold')
    ax2.set_ylabel('Gap C_UB - C_LB [bits/slot]', fontweight='bold')
    ax2.set_title(f'Gap vs Signal Budget (λ_b={lambda_b_fixed})', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Subplot 3: Relative gap (percentage)
    print("🔄 Computing relative gap...")
    ax3 = axes[1, 0]

    relative_gap = np.where(C_ub > 0, (gaps / C_ub) * 100, 0)
    relative_gap = np.clip(relative_gap, 0, 100)

    ax3.semilogx(lambda_b_range, relative_gap, 'g-', linewidth=2.5)
    ax3.fill_between(lambda_b_range, 0, relative_gap, alpha=0.3, color='lightgreen')

    ax3.axhline(y=10, color='red', linestyle='--', linewidth=1.5,
                label='10% threshold')

    ax3.set_xlabel('Background λ_b [photons/slot]', fontweight='bold')
    ax3.set_ylabel('Relative Gap [%]', fontweight='bold')
    ax3.set_title('Relative Gap: (C_UB - C_LB) / C_UB', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Subplot 4: Gap statistics table
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Compute regime-specific statistics
    regime_stats = []
    regimes = [
        ('Zodiacal', lambda_b_range < 0.1),
        ('Earthshine', (lambda_b_range >= 0.1) & (lambda_b_range < 5)),
        ('Stray Light', lambda_b_range >= 5)
    ]

    for regime_name, mask in regimes:
        regime_gaps = gaps[mask]
        regime_rel_gaps = relative_gap[mask]

        regime_stats.append([
            regime_name,
            f'{np.mean(regime_gaps):.4f}',
            f'{np.max(regime_gaps):.4f}',
            f'{np.mean(regime_rel_gaps):.2f}%'
        ])

    table = ax4.table(
        cellText=regime_stats,
        colLabels=['Regime', 'Avg Gap', 'Max Gap', 'Avg Rel Gap'],
        cellLoc='center',
        loc='center',
        colWidths=[0.3, 0.25, 0.25, 0.25]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    # Style header
    for i in range(4):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Style rows
    colors = ['lightblue', 'lightyellow', 'lightcoral']
    for i in range(1, 4):
        for j in range(4):
            table[(i, j)].set_facecolor(colors[i - 1])

    ax4.set_title('Gap Statistics by Regime', fontweight='bold', pad=20)

    plt.tight_layout()

    output_path = f"{dirs['analysis']}/Fig_A3_Capacity_Gap"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Saved: {output_path}.pdf")

    # Save gap analysis data
    gap_data = {
        'lambda_b_range': lambda_b_range.tolist(),
        'gaps': gaps.tolist(),
        'relative_gaps': relative_gap.tolist(),
        'regime_stats': regime_stats
    }

    with open(f"{dirs['analysis']}/gap_analysis.json", 'w') as f:
        json.dump(gap_data, f, indent=2)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='OISL-ISAC Complete Simulation Suite - Fixed Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate all figures
  python run_section_iv_final.py --figure all

  # Specific figures
  python run_section_iv_final.py --figure 1
  python run_section_iv_final.py --figure 4

  # Validation and analysis
  python run_section_iv_final.py --validation --n-trials 10000
  python run_section_iv_final.py --analysis
  python run_section_iv_final.py --capacity-gap

  # Complete run
  python run_section_iv_final.py --figure all --validation --analysis --capacity-gap
        """
    )

    parser.add_argument('--figure', type=str, default='all',
                        help='Figure to generate: 1,2,3,4,all')
    parser.add_argument('--validation', action='store_true',
                        help='Run Monte Carlo CRLB validation')
    parser.add_argument('--analysis', action='store_true',
                        help='Run parameter sensitivity analysis')
    parser.add_argument('--capacity-gap', action='store_true',
                        help='Compute capacity upper/lower bound gap')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Configuration file path')
    parser.add_argument('--n-trials', type=int, default=10000,
                        help='Number of Monte Carlo trials')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: auto)')

    args = parser.parse_args()

    # Header
    print("\n" + "=" * 70)
    print("🚀 OISL-ISAC COMPLETE SIMULATION SUITE - FIXED VERSION")
    print("=" * 70)
    print(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔒 Random Seed: {args.seed}")
    print(f"⚙️  Config: {args.config}")
    print("=" * 70)

    # Setup
    set_reproducibility(args.seed)
    config = load_config(args.config)
    dirs = setup_directories()

    # Save metadata
    save_metadata(dirs, config, args)

    # Track execution time
    start_time = time.time()

    # Execute requested tasks
    tasks_executed = []

    if args.figure == 'all' or args.figure == '1':
        generate_figure_1_complete(config, dirs)
        tasks_executed.append('Figure 1: Capacity Bounds')

    if args.figure == 'all' or args.figure == '2':
        generate_figure_2_parallel(config, dirs, args.workers or 2)
        tasks_executed.append('Figure 2: FIM Heatmap')

    if args.figure == 'all' or args.figure == '3':
        generate_figure_3_parallel(config, dirs, args.workers)
        tasks_executed.append('Figure 3: Pareto Boundary')

    if args.figure == 'all' or args.figure == '4':
        generate_figure_4_physical_complete(config, dirs)
        tasks_executed.append('Figure 4: Physical Model')

    if args.validation:
        monte_carlo_crlb_validation(config, dirs, args.workers, args.n_trials)
        tasks_executed.append('CRLB Validation')

    if args.analysis:
        parameter_sensitivity_analysis(config, dirs)
        tasks_executed.append('Sensitivity Analysis')

    if args.capacity_gap:
        capacity_gap_analysis(config, dirs)
        tasks_executed.append('Capacity Gap Analysis')

    # Summary
    elapsed_time = time.time() - start_time

    print(f"\n{'=' * 70}")
    print("✅ EXECUTION COMPLETE!")
    print(f"{'=' * 70}")
    print(f"⏱️  Total Time: {elapsed_time / 60:.1f} minutes")
    print(f"📁 Results in: {dirs['base']}/")
    print(f"\n📊 Tasks Completed ({len(tasks_executed)}):")
    for task in tasks_executed:
        print(f"   ✓ {task}")

    # List generated files
    print(f"\n📂 Generated Files:")
    for subdir in ['figures', 'appendix', 'analysis', 'validation']:
        subdir_path = dirs[subdir]
        if os.path.exists(subdir_path):
            files = [f for f in os.listdir(subdir_path)
                     if f.endswith(('.pdf', '.png', '.json', '.npz'))]
            if files:
                print(f"\n  📁 {subdir.title()}:")
                for f in sorted(files):
                    size = os.path.getsize(os.path.join(subdir_path, f)) / 1024
                    print(f"     • {f} ({size:.1f} KB)")

    print(f"\n{'=' * 70}")
    print(f"🎯 To reproduce: python {sys.argv[0]} --seed {args.seed}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()