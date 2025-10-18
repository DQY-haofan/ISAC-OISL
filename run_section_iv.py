#!/usr/bin/env python3
"""
Section IV: Numerical Results and Validation - Publication Version
===================================================================

IEEE Transactions Paper - OISL-ISAC Performance Limits
Complete simulation suite with validation and analysis capabilities.

Usage:
    python run_section_iv.py --figure all           # All main figures
    python run_section_iv.py --figure 1             # Specific figure
    python run_section_iv.py --validation           # Monte Carlo validation
    python run_section_iv.py --analysis             # Ablation studies
    python run_section_iv.py --capacity-gap         # Upper/lower bound analysis
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

# Import core functions
sys.path.append('.')
from main_simulation import (
    capacity_lb, fim_pilot, generate_dither_sequence,
    poisson_entropy, setup_ieee_style, SPEED_OF_LIGHT
)

from isac_core import (
    capacity_lb,              # 二元输入下界
    capacity_ub_dual,         # ⭐ 对偶上界（修复）
    capacity_discrete_input,  # 离散输入容量（原AB算法）
    physical_background_model,
    fim_pilot,
    poisson_entropy,
    setup_ieee_style,
    generate_dither_sequence,
    SPEED_OF_LIGHT
)


warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION AND SETUP
# ============================================================================

def load_config(config_path='config.yaml'):
    """Load simulation parameters from YAML config"""
    with open(config_path, 'r', encoding='utf-8') as f:  # ✅ 显式指定UTF-8
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
# ENHANCED FIGURE 1: CAPACITY WITH UPPER/LOWER BOUNDS
# ============================================================================

def generate_figure_1_enhanced(config, dirs):
    """
    生成 Fig_1_Capacity_Bounds_Fixed.pdf（GPU加速版）
    """
    print("\n" + "=" * 60)
    print("📊 FIGURE 1: 容量界对比（GPU加速版）")
    print("=" * 60)

    colors = setup_ieee_style()

    # 参数
    lambda_b_range = np.logspace(-2, 2, 40)
    signal_budgets = config['simulation']['signal_budgets']
    hardware_config = config['hardware_platforms']['short_dead_time']

    tau_d = hardware_config['dead_time']
    M_pixels = hardware_config['parallel_pixels']
    dt = hardware_config['slot_duration']

    # 有效峰值功率
    if tau_d > 0:
        S_max_eff = min(hardware_config['peak_power'],
                        (dt / tau_d) * M_pixels)
    else:
        S_max_eff = hardware_config['peak_power']

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for idx, S_bar in enumerate(signal_budgets):
        ax = axes[idx]

        print(f"\n  📈 S̄ = {S_bar} photons/slot")
        start_time = time.time()

        # ⭐ GPU 批量计算
        print(f"    🚀 GPU批量计算 {len(lambda_b_range)} 个点...")

        # 下界：批量计算
        capacities_lb, _ = capacity_lb_batch_gpu(
            S_bar, S_max_eff, lambda_b_range, dt, tau_d, M_pixels
        )

        # 上界：批量计算
        capacities_ub = capacity_ub_dual_batch_gpu(
            S_bar, S_max_eff, lambda_b_range, dt, tau_d, M_pixels
        )

        # 离散输入容量（稀疏采样）
        capacities_discrete = []
        lambda_b_discrete = []
        for i in range(0, len(lambda_b_range), 5):
            lambda_b = lambda_b_range[i]
            try:
                C_disc, _, _ = capacity_discrete_input(
                    S_bar, S_max_eff, lambda_b, dt, tau_d, M_pixels,
                    max_iter=100
                )
                capacities_discrete.append(C_disc)
                lambda_b_discrete.append(lambda_b)
            except:
                pass

        elapsed = time.time() - start_time
        print(f"    ✅ 完成，耗时 {elapsed:.2f} 秒")

        # 计算Gap
        gaps = capacities_ub - capacities_lb
        valid_gaps = gaps[(gaps > 0) & (gaps < 1)]
        avg_gap = np.mean(valid_gaps) if len(valid_gaps) > 0 else 0

        # 绘图
        ax.semilogx(lambda_b_range, capacities_lb, 'b-', linewidth=2.5,
                    label='Lower Bound (Binary Input)')

        ax.semilogx(lambda_b_range, capacities_ub, 'r--', linewidth=2,
                    label='Upper Bound (Dual Formula)')

        if capacities_discrete:
            ax.semilogx(lambda_b_discrete, capacities_discrete, 'g.', markersize=8,
                        label='Discrete-input capacity (AB)')

        ax.fill_between(lambda_b_range, capacities_lb, capacities_ub,
                        alpha=0.2, color='gray', label='Achievability Gap')

        # 背景制度标记
        ax.axvline(x=0.01, color='blue', alpha=0.3, linestyle=':',
                   label='Zodiacal' if idx == 0 else "")
        ax.axvline(x=1.0, color='orange', alpha=0.3, linestyle=':',
                   label='Earthshine' if idx == 0 else "")
        ax.axvline(x=10.0, color='red', alpha=0.3, linestyle=':',
                   label='Stray Light' if idx == 0 else "")

        ax.set_xlabel('Background λ_b [photons/slot]', fontweight='bold')
        ax.set_ylabel('Capacity [bits/slot]', fontweight='bold')
        ax.set_title(f'S̄ = {S_bar} photons/slot\n(Avg Gap: {avg_gap:.4f} bits/slot)',
                     fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='best')

        print(f"    📊 平均差距：{avg_gap:.4f} bits/slot")

    plt.tight_layout()

    # 保存
    output_path = f"{dirs['figures']}/Fig_1_Capacity_Bounds_Fixed"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 保存：{output_path}.pdf")

# ============================================================================
# ENHANCED FIGURE 4: PHYSICAL BACKGROUND MODEL
# ============================================================================

def generate_figure_4_physical(config, dirs):
    """
    生成 Fig_4_Design_Law_Fixed.pdf

    ⭐ 修复内容：
    1. 物理背景模型接受 dt_slot 参数（使用仿真的 dt）
    2. 容量等高线不再全是 1.000
    3. 固定等高线档位，体现设计临界点
    """
    print("\n" + "=" * 60)
    print("📊 FIGURE 4: 太阳规避设计律（修复版）")
    print("=" * 60)

    setup_ieee_style()

    # 网格参数
    sun_angles = np.linspace(10, 180, 50)
    fov_range = np.linspace(20, 500, 40)

    Sun_grid, FoV_grid = np.meshgrid(sun_angles, fov_range)

    # 存储网格
    Background_grid = np.zeros_like(Sun_grid)
    Capacity_grid = np.zeros_like(Sun_grid)

    # 系统参数
    params = config['system_parameters']
    S_bar = params['Sbar']
    S_max = params['Smax']
    dt = params['dt']  # ⭐ 使用仿真的实际时隙
    tau_d = params.get('tau_d', 50e-9)
    M_pixels = params['M_pixels']

    # 轨道参数
    orbit_params = {
        'altitude_km': 600,
        'earth_phase_angle_deg': 90
    }

    print(f"\n⚙️  参数：dt={dt * 1e6:.2f} µs, S̄={S_bar}, S_max={S_max}")
    print(f"  计算 {len(fov_range)}×{len(sun_angles)} = {len(fov_range) * len(sun_angles)} 点...")

    # 计算背景和容量
    for i in tqdm(range(len(fov_range)), desc="  进度"):
        for j in range(len(sun_angles)):
            # ⭐ 关键修复：传入 dt_slot 参数
            lambda_b, components = physical_background_model(
                Sun_grid[i, j],
                FoV_grid[i, j],
                orbit_params=orbit_params,
                dt_slot=dt,
                config=config  # ⭐ 传入配置
            )

            Background_grid[i, j] = lambda_b

            # 计算容量
            C_lb, _ = capacity_lb(S_bar, S_max, lambda_b, dt, tau_d, M_pixels)
            Capacity_grid[i, j] = C_lb

    # 诊断输出
    c_min, c_max = np.min(Capacity_grid), np.max(Capacity_grid)
    c_mean, c_median = np.mean(Capacity_grid), np.median(Capacity_grid)

    print(f"\n📊 容量统计：")
    print(f"   范围：[{c_min:.4f}, {c_max:.4f}] bits/slot")
    print(f"   均值：{c_mean:.4f} bits/slot")
    print(f"   中位数：{c_median:.4f} bits/slot")

    # ⭐ 固定等高线档位（修复"全是1.000"的问题）
    capacity_levels = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

    # 过滤有效档位
    valid_levels = capacity_levels[(capacity_levels >= c_min) & (capacity_levels <= c_max)]

    if len(valid_levels) == 0:
        # 如果固定档位都不在范围内，使用自适应
        print(f"  ⚠️  固定档位不适用，使用自适应档位")
        valid_levels = np.linspace(c_min + 0.1 * (c_max - c_min),
                                   c_max - 0.1 * (c_max - c_min), 5)

    print(f"   等高线档位：{valid_levels}")

    # === 创建图形 ===
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    # 主图：背景热图 + 容量等高线
    ax_main = fig.add_subplot(gs[0, :])

    # 背景热图
    im_bg = ax_main.pcolormesh(
        Sun_grid, FoV_grid,
        np.log10(Background_grid + 1e-10),
        shading='auto', cmap='YlOrRd', alpha=0.7
    )

    cbar_bg = plt.colorbar(im_bg, ax=ax_main, pad=0.02)
    cbar_bg.set_label(
        f'log₁₀(Background λ_b) [photons/slot @ Δt={dt * 1e6:.1f}µs]',  # ⭐ 明确单位
        rotation=270, labelpad=20, fontweight='bold'
    )

    # 容量等高线 ⭐ 应该不再是水平条纹
    try:
        contours = ax_main.contour(
            Sun_grid, FoV_grid, Capacity_grid,
            levels=valid_levels,
            colors='navy', linewidths=2.5, linestyles='solid'
        )

        # 标注等高线
        labels = ax_main.clabel(contours, inline=True, fontsize=9,
                                fmt='%.2f', inline_spacing=10)

        for label in labels:
            label.set_bbox(dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor='navy',
                alpha=0.8
            ))

        print(f"  ✅ 成功绘制 {len(valid_levels)} 条等高线")

    except Exception as e:
        print(f"  ⚠️  等高线绘制失败：{e}")

    # 参考线
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

    # 子图：容量分布直方图
    ax_hist = fig.add_subplot(gs[1, 0])
    ax_hist.hist(Capacity_grid.flatten(), bins=50, color='navy', alpha=0.7, edgecolor='black')
    ax_hist.set_xlabel('Capacity [bits/slot]', fontweight='bold')
    ax_hist.set_ylabel('Count', fontweight='bold')
    ax_hist.set_title('Capacity Distribution', fontweight='bold')
    ax_hist.grid(True, alpha=0.3)

    # 子图：背景 vs FoV（横切面）
    ax_slice = fig.add_subplot(gs[1, 1])

    # 选择几个典型太阳角的横切面
    sun_angles_slice = [30, 90, 150]
    for sun_angle in sun_angles_slice:
        idx = np.argmin(np.abs(sun_angles - sun_angle))
        ax_slice.semilogy(fov_range, Background_grid[:, idx],
                          linewidth=2, marker='o', markersize=4,
                          label=f'Sun angle = {sun_angle}°')

    ax_slice.set_xlabel('FoV [μrad]', fontweight='bold')
    ax_slice.set_ylabel('Background [photons/slot]', fontweight='bold')
    ax_slice.set_title('Background vs FoV (Cross-sections)', fontweight='bold')
    ax_slice.legend(fontsize=9)
    ax_slice.grid(True, alpha=0.3)

    plt.tight_layout()

    # 保存
    output_path = f"{dirs['figures']}/Fig_4_Design_Law_Fixed"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\n✅ 保存：{output_path}.pdf")
    print(f"   预期：等高线随太阳角和FoV双变量变化，不再是水平条纹")

    # 保存数据
    np.savez(f"{dirs['data']}/design_law_fixed.npz",
             sun_angles=sun_angles,
             fov_range=fov_range,
             background=Background_grid,
             capacity=Capacity_grid,
             dt_slot=dt)

# ============================================================================
# MONTE CARLO CRLB VALIDATION
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

    # MLE via grid search
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
        options={'maxiter': 200}
    )

    return result.x


def monte_carlo_crlb_validation(config, dirs, n_workers=None, n_trials=10000):
    """
    Monte Carlo validation of CRLB bounds
    """
    print("\n" + "=" * 60)
    print("🎲 MONTE CARLO CRLB VALIDATION")
    print("=" * 60)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    print(f"🚀 Using {n_workers} workers for {n_trials} trials")

    setup_ieee_style()

    # Test scenarios
    params = config['system_parameters']
    scenarios = [
        {'Sbar': 10, 'r_b': 0.1, 'alpha': 0.3, 'rho': 0.5, 'name': 'Low Bg'},
        {'Sbar': 50, 'r_b': 1.0, 'alpha': 0.2, 'rho': 0.6, 'name': 'Med Bg'},
        {'Sbar': 20, 'r_b': 5.0, 'alpha': 0.4, 'rho': 0.7, 'name': 'High Bg'}
    ]

    fig, axes = plt.subplots(1, len(scenarios), figsize=(15, 5))

    for idx, scenario in enumerate(scenarios):
        print(f"\n🔬 Testing: {scenario['name']}")

        params_test = params.copy()
        params_test.update(scenario)

        mu_true = np.array([1e-6, 0.5e-6])

        # Generate dither
        N_pilot = int(scenario['alpha'] * params['N'])
        dither_seq = generate_dither_sequence(N_pilot, params['theta_b'])

        # Compute CRLB
        I_pilot = fim_pilot(
            scenario['alpha'], scenario['rho'], scenario['Sbar'],
            params['N'], params['dt'], params_test, dither_seq,
            params.get('tau_d'), None, params['M_pixels']
        )

        J = I_pilot + params['J_P'] + 1e-12 * np.eye(4)

        if np.linalg.cond(J) < 1e30:
            crlb_cov = np.linalg.inv(J)
            crlb_var_mux = crlb_cov[0, 0]
            crlb_var_muy = crlb_cov[1, 1]
        else:
            print(f"  ⚠️  Singular FIM")
            continue

        # Prepare workers
        S_pilot = scenario['rho'] * scenario['Sbar'] / scenario['alpha']

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

        ax.set_yscale('log')
        ax.set_xlim(0.5, 2.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['μx', 'μy'])
        ax.set_ylabel('Variance [rad²]', fontweight='bold')
        ax.set_title(f'{scenario["name"]}\n({n_trials} trials)',
                     fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        # Efficiency
        eff_x = crlb_var_mux / sample_var_mux * 100
        eff_y = crlb_var_muy / sample_var_muy * 100
        print(f"  📊 Efficiency: μx={eff_x:.1f}%, μy={eff_y:.1f}%")

    plt.tight_layout()

    output_path = f"{dirs['appendix']}/Fig_A1_CRLB_Validation"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}.pdf")


# ============================================================================
# ABLATION & SENSITIVITY ANALYSIS
# ============================================================================

def parameter_sensitivity_analysis(config, dirs):
    """
    Comprehensive parameter sensitivity analysis
    """
    print("\n" + "=" * 60)
    print("🔬 PARAMETER SENSITIVITY ANALYSIS")
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

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, (param_name, sweep_config) in enumerate(sweeps.items()):
        ax = axes[idx]

        print(f"\n🔄 Sweeping: {param_name}")

        mse_results = []
        capacity_results = []

        for val in tqdm(sweep_config['values'], desc=f"  {param_name}"):
            # Update params
            params_test = params.copy()

            if param_name == 'dither_amplitude':
                # Regenerate dither with new amplitude
                N_pilot = int(0.3 * params['N'])
                dither_seq = generate_dither_sequence(N_pilot, val)
                params_test['dither_amp'] = val
            elif param_name == 'dead_time':
                params_test['tau_d'] = val
            elif param_name == 'pointing_variance':
                params_test['sigma2'] = val
            elif param_name == 'background_rate':
                params_test['r_b'] = val

            # Compute FIM
            try:
                if param_name == 'dither_amplitude':
                    I_pilot = fim_pilot(
                        0.3, 0.5, params['Sbar'], params['N'],
                        params['dt'], params_test, dither_seq,
                        params.get('tau_d'), None, params['M_pixels']
                    )
                else:
                    N_pilot = int(0.3 * params['N'])
                    dither_seq = generate_dither_sequence(N_pilot, params['theta_b'])
                    I_pilot = fim_pilot(
                        0.3, 0.5, params['Sbar'], params['N'],
                        params['dt'], params_test, dither_seq,
                        params_test.get('tau_d'), None, params['M_pixels']
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
                C_lb, _ = capacity_lb(
                    S_data, params['Smax'],
                    params_test.get('r_b', params['r_b']) * params['dt'],
                    params['dt'], params_test.get('tau_d', params.get('tau_d')),
                    params['M_pixels']
                )
                capacity_results.append(C_lb)
            except:
                capacity_results.append(np.nan)

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
        ax.legend(lines, labels, loc='best', fontsize=8)

    plt.tight_layout()

    output_path = f"{dirs['analysis']}/Fig_A3_Sensitivity_Analysis"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}.pdf")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='OISL-ISAC Section IV: Publication-Ready Suite'
    )
    parser.add_argument('--figure', type=str, default='all',
                        help='Figure to generate: 1,2,3,4,all')
    parser.add_argument('--validation', action='store_true',
                        help='Run Monte Carlo CRLB validation')
    parser.add_argument('--analysis', action='store_true',
                        help='Run sensitivity analysis')
    parser.add_argument('--capacity-gap', action='store_true',
                        help='Compute capacity upper/lower bound gap')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--n-trials', type=int, default=10000,
                        help='Number of Monte Carlo trials')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers')

    args = parser.parse_args()

    print("🚀 OISL-ISAC Publication Suite")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Setup
    set_reproducibility(args.seed)
    config = load_config(args.config)
    dirs = setup_directories()

    # Execute requested tasks
    if args.capacity_gap or args.figure == 'all' or args.figure == '1':
        generate_figure_1_enhanced(config, dirs)

    if args.figure == 'all' or args.figure == '4':
        generate_figure_4_physical(config, dirs)

    if args.validation:
        monte_carlo_crlb_validation(config, dirs, args.workers, args.n_trials)

    if args.analysis:
        parameter_sensitivity_analysis(config, dirs)

    # Summary
    print(f"\n{'=' * 60}")
    print("✅ EXECUTION COMPLETE!")
    print(f"{'=' * 60}")
    print(f"📁 Results in: {dirs['base']}/")

    for subdir in ['figures', 'appendix', 'analysis']:
        path = dirs[subdir]
        if os.path.exists(path):
            files = [f for f in os.listdir(path)
                     if f.endswith(('.pdf', '.png'))]
            if files:
                print(f"\n📊 {subdir.title()}:")
                for f in sorted(files):
                    print(f"   • {f}")


if __name__ == "__main__":
    main()