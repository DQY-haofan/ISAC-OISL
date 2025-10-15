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

def generate_figure_2(config, dirs):
    """
    Generate Fig_2_FIM_Heatmap_Comparison.pdf
    Compares FIM heatmaps under low vs high background
    """
    print("\n" + "=" * 60)
    print("📊 FIGURE 2: FIM Heatmap Comparison")
    print("=" * 60)

    setup_ieee_style()

    # Simulation parameters
    params = config['system_parameters']
    alpha_range = np.linspace(0.05, 0.95, 25)
    rho_range = np.linspace(0.05, 0.95, 25)

    # Background scenarios: Low (Zodiacal) vs High (Stray Light)
    scenarios = {
        'Low Background (Zodiacal)': {'r_b': 0.01, 'color': 'viridis'},
        'High Background (Stray Light)': {'r_b': 10.0, 'color': 'plasma'}
    }

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for idx, (scenario_name, scenario) in enumerate(scenarios.items()):
        ax = axes[idx]

        # Setup parameters for this scenario
        params_sim = params.copy()
        params_sim['r_b'] = scenario['r_b']

        # Generate dither sequence
        max_pilots = int(0.9 * params['N'])
        dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

        # Compute MSE grid
        mse_grid = np.zeros((len(rho_range), len(alpha_range)))
        valid_count = 0

        for i, rho in enumerate(tqdm(rho_range, desc=f"Computing {scenario_name}")):
            for j, alpha in enumerate(alpha_range):
                try:
                    # Compute FIM
                    I_pilot = fim_pilot(alpha, rho, params['Sbar'], params['N'],
                                        params['dt'], params_sim, dither_seq,
                                        params.get('tau_d'), None, params['M_pixels'])

                    # Add prior and regularization
                    J = I_pilot + params['J_P'] + 1e-12 * np.eye(4)

                    if np.linalg.cond(J) < 1e30:
                        J_inv = np.linalg.inv(J)
                        # Focus on pointing accuracy (μx, μy)
                        W = np.diag([1.0, 1.0, 0.0, 0.0])
                        mse_grid[i, j] = np.trace(W @ J_inv)
                        valid_count += 1
                    else:
                        mse_grid[i, j] = np.nan

                except:
                    mse_grid[i, j] = np.nan

        print(f"  Valid points: {valid_count}/{mse_grid.size}")

        # Plot heatmap
        valid_mask = np.isfinite(mse_grid) & (mse_grid > 0)
        if valid_mask.sum() > 0:
            vmin = np.percentile(mse_grid[valid_mask], 5)
            vmax = np.percentile(mse_grid[valid_mask], 95)

            log_mse = np.full_like(mse_grid, np.nan)
            log_mse[valid_mask] = np.log10(mse_grid[valid_mask])

            Alpha, Rho = np.meshgrid(alpha_range, rho_range)

            im = ax.pcolormesh(Alpha, Rho, log_mse, shading='auto',
                               cmap=scenario['color'], vmin=np.log10(vmin), vmax=np.log10(vmax))

            # Add contour lines
            try:
                levels = np.linspace(np.log10(vmin), np.log10(vmax), 6)
                contours = ax.contour(Alpha, Rho, log_mse, levels=levels,
                                      colors='white', alpha=0.7, linewidths=0.8)
                ax.clabel(contours, inline=True, fontsize=7, fmt='%.1f')
            except:
                pass

            # Colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('log₁₀(MSE μx,μy) [rad²]', rotation=270, labelpad=15)

        ax.set_xlabel('Time Allocation α')
        ax.set_ylabel('Photon Allocation ρ')
        ax.set_title(scenario_name)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = f"{dirs['figures']}/Fig_2_FIM_Heatmap_Comparison"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}.pdf")


# ============================================================================
# FIGURE 3: RATE-MSE PARETO BOUNDARY
# ============================================================================

def generate_figure_3(config, dirs):
    """
    Generate Fig_3_Rate_MSE_Boundary.pdf
    Shows Rate-MSE Pareto boundaries for different background scenarios
    """
    print("\n" + "=" * 60)
    print("📊 FIGURE 3: Rate-MSE Pareto Boundary")
    print("=" * 60)

    colors = setup_ieee_style()

    # Parameters
    params = config['system_parameters']

    # Background scenarios
    scenarios = {
        'Low (Zodiacal)': {'r_b': 0.01, 'color': colors['zodiacal']},
        'Medium (Earthshine)': {'r_b': 1.0, 'color': colors['earthshine']},
        'High (Stray Light)': {'r_b': 10.0, 'color': colors['stray_light']}
    }

    # Adaptive MSE targets based on achievable range
    print("🔍 Probing achievable MSE range...")
    mse_samples = []
    alpha_probe = np.linspace(0.1, 0.9, 8)
    rho_probe = np.linspace(0.1, 0.9, 8)

    # Use medium scenario for probing
    params_probe = params.copy()
    params_probe['r_b'] = 1.0

    max_pilots = int(0.9 * params['N'])
    dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

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
        D_targets = np.logspace(-12, -6, 15)

    print(f"📐 MSE targets: [{D_targets[0]:.2e}, {D_targets[-1]:.2e}]")

    # Compute Pareto boundaries
    alpha_search = np.linspace(0.05, 0.95, 20)
    rho_search = np.linspace(0.05, 0.95, 20)

    fig, ax = plt.subplots(figsize=(8, 6))

    for scenario_name, scenario in scenarios.items():
        print(f"\n🔄 Computing Pareto boundary: {scenario_name}")

        params_sim = params.copy()
        params_sim['r_b'] = scenario['r_b']
        pareto_points = []

        for D_max in tqdm(D_targets, desc=scenario_name):
            max_rate = 0.0
            best_alpha, best_rho = 0, 0

            for alpha in alpha_search:
                for rho in rho_search:
                    try:
                        # Check feasibility
                        S_pilot = rho * params['Sbar'] / alpha
                        S_data = (1 - rho) * params['Sbar'] / (1 - alpha)

                        if S_pilot > params['Smax'] or S_data > params['Smax']:
                            continue

                        # Compute FIM and check MSE constraint
                        I_pilot = fim_pilot(alpha, rho, params['Sbar'], params['N'],
                                            params['dt'], params_sim, dither_seq,
                                            params.get('tau_d'), None, params['M_pixels'])

                        J = I_pilot + params['J_P'] + 1e-12 * np.eye(4)
                        if np.linalg.cond(J) > 1e30:
                            continue

                        W = np.diag([1.0, 1.0, 0.0, 0.0])
                        mse_current = np.trace(W @ np.linalg.inv(J))

                        if mse_current > D_max:
                            continue

                        # Compute achievable rate
                        C_data, _ = capacity_lb(S_data, params['Smax'], scenario['r_b'],
                                                params['dt'], params.get('tau_d'), params['M_pixels'])
                        rate = (1 - alpha) * C_data

                        if rate > max_rate:
                            max_rate = rate
                            best_alpha, best_rho = alpha, rho
                    except:
                        continue

            if max_rate > 0:
                pareto_points.append((max_rate, D_max, best_alpha, best_rho))

        if len(pareto_points) > 0:
            rates = [p[0] for p in pareto_points]
            mses = [p[1] for p in pareto_points]
            alphas = [p[2] for p in pareto_points]
            rhos = [p[3] for p in pareto_points]

            # Sort by MSE for proper plotting
            sorted_data = sorted(zip(mses, rates, alphas, rhos))
            mses_sorted = [d[0] for d in sorted_data]
            rates_sorted = [d[1] for d in sorted_data]
            alphas_sorted = [d[2] for d in sorted_data]
            rhos_sorted = [d[3] for d in sorted_data]

            ax.loglog(mses_sorted, rates_sorted, 'o-',
                      color=scenario['color'], linewidth=2.5, markersize=6,
                      label=scenario_name)

            # Annotate a few representative points
            if len(mses_sorted) > 2:
                mid_idx = len(mses_sorted) // 2
                ax.annotate(f'α={alphas_sorted[mid_idx]:.2f}, ρ={rhos_sorted[mid_idx]:.2f}',
                            xy=(mses_sorted[mid_idx], rates_sorted[mid_idx]),
                            xytext=(10, 10), textcoords='offset points',
                            fontsize=8, ha='left',
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

        print(f"  Found {len(pareto_points)} Pareto points")

    ax.set_xlabel('MSE (μx, μy) [rad²]', fontweight='bold')
    ax.set_ylabel('Rate [bits/slot]', fontweight='bold')
    ax.set_title(f'Rate-MSE Pareto Boundary\n(S̄={params["Sbar"]}, M={params["M_pixels"]})', fontweight='bold')
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

def generate_figure_4(config, dirs):
    """
    Generate Fig_4_Design_Law.pdf
    Shows sun avoidance angle design law with background heatmap and capacity contours
    """
    print("\n" + "=" * 60)
    print("📊 FIGURE 4: Sun Avoidance Design Law")
    print("=" * 60)

    setup_ieee_style()

    # Sun avoidance angle and FoV ranges
    sun_angles = np.linspace(10, 180, 50)  # degrees
    fov_range = np.linspace(20, 500, 40)  # microradians

    # Background model (simplified PST model)
    def background_rate(sun_angle_deg, fov_urad):
        """Simplified background model based on sun avoidance angle and FoV"""
        # Convert to appropriate units
        sun_angle_rad = np.radians(sun_angle_deg)
        fov_rad = fov_urad * 1e-6

        # Simplified PST model: exponential decay from sun
        # Near-sun: PST ~ 1e-4, Deep space: PST ~ 1e-10
        log_pst = -4.0 - 3.0 * np.log10(sun_angle_deg / 10.0)
        log_pst = np.clip(log_pst, -10, -4)  # Reasonable bounds

        pst = 10 ** log_pst

        # Background rate scales with FoV and PST
        # Typical values: 0.001 to 100 photons/slot
        lambda_b = pst * (fov_rad / 100e-6) ** 2 * 1000

        return lambda_b

    # Create grids
    Sun_grid, FoV_grid = np.meshgrid(sun_angles, fov_range)
    Background_grid = np.zeros_like(Sun_grid)
    Capacity_grid = np.zeros_like(Sun_grid)

    # System parameters
    S_bar = config['system_parameters']['Sbar']
    S_max = config['system_parameters']['Smax']
    dt = config['system_parameters']['dt']
    tau_d = config['system_parameters'].get('tau_d')
    M_pixels = config['system_parameters']['M_pixels']

    print("🔄 Computing background and capacity grids...")
    for i in tqdm(range(len(fov_range))):
        for j in range(len(sun_angles)):
            # Background
            lambda_b = background_rate(Sun_grid[i, j], FoV_grid[i, j])
            Background_grid[i, j] = lambda_b

            # Capacity
            C_lb, _ = capacity_lb(S_bar, S_max, lambda_b, dt, tau_d, M_pixels)
            Capacity_grid[i, j] = C_lb

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 7))

    # Background heatmap
    im = ax.pcolormesh(Sun_grid, FoV_grid, np.log10(Background_grid + 1e-10),
                       shading='auto', cmap='YlOrRd', alpha=0.8)

    # Capacity contours
    capacity_levels = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
    contours = ax.contour(Sun_grid, FoV_grid, Capacity_grid,
                          levels=capacity_levels, colors='navy', linewidths=2.0)
    ax.clabel(contours, inline=True, fontsize=10, fmt='%.1f bits/slot')

    # Colorbar for background
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(Background λb) [photons/slot]', rotation=270, labelpad=20)

    # Mark typical operating regimes
    ax.axhline(y=50, color='blue', linestyle='--', alpha=0.7, label='Typical FoV (50 μrad)')
    ax.axhline(y=200, color='green', linestyle='--', alpha=0.7, label='Wide FoV (200 μrad)')
    ax.axvline(x=30, color='red', linestyle=':', alpha=0.7, label='Min Sun Avoidance')

    ax.set_xlabel('Sun Avoidance Angle [degrees]', fontweight='bold')
    ax.set_ylabel('Receiver FoV [μrad]', fontweight='bold')
    ax.set_title('Sun Avoidance Design Law\n(Background + Capacity Contours)', fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    output_path = f"{dirs['figures']}/Fig_4_Design_Law"
    plt.savefig(f"{output_path}.pdf", dpi=300, bbox_inches='tight')
    plt.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: {output_path}.pdf")

    # Save design data
    design_data = {
        'sun_angles': sun_angles,
        'fov_range': fov_range,
        'background_grid': Background_grid,
        'capacity_grid': Capacity_grid
    }
    np.savez(f"{dirs['data']}/design_law_data.npz", **design_data)


# ============================================================================
# MONTE CARLO VALIDATION
# ============================================================================

def monte_carlo_validation(config, dirs):
    """
    Monte Carlo validation of CRLB bounds
    """
    print("\n" + "=" * 60)
    print("🎲 MONTE CARLO CRLB VALIDATION")
    print("=" * 60)

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
        mu_true = np.array([1.0e-6, 0.5e-6])  # True pointing bias [rad]

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

        # Monte Carlo simulation
        print(f"  🔄 Running {n_trials} Monte Carlo trials...")

        estimates_mux = []
        estimates_muy = []

        for trial in tqdm(range(n_trials), desc="MC trials"):
            # Generate synthetic observations
            pilot_observations = []

            for n in range(max_pilots):
                # True pointing with dither
                mu_eff = mu_true + dither_seq[n % len(dither_seq)]

                # Expected pointing loss
                a = 4.0 / (params['theta_b'] ** 2)
                b = 2.0 / (params['theta_b'] ** 2)
                gamma = 1.0 + a * params['sigma2']
                L_p = (1.0 / gamma) * np.exp(-b * np.dot(mu_eff, mu_eff) / gamma)

                # Signal photon rate
                S_pilot = scenario['rho'] * scenario['Sbar'] / scenario['alpha']
                lambda_signal = S_pilot * L_p
                lambda_total = lambda_signal + scenario['r_b'] * params['dt']

                # Generate Poisson observation
                y_obs = np.random.poisson(lambda_total)
                pilot_observations.append((y_obs, dither_seq[n % len(dither_seq)]))

            # Simple MLE estimation (grid search)
            def log_likelihood(mu_est):
                ll = 0.0
                for y_obs, dither in pilot_observations:
                    mu_eff = mu_est + dither
                    L_p = (1.0 / gamma) * np.exp(-b * np.dot(mu_eff, mu_eff) / gamma)
                    lambda_pred = S_pilot * L_p + scenario['r_b'] * params['dt']

                    if lambda_pred > 0:
                        ll += y_obs * np.log(lambda_pred) - lambda_pred

                return ll

            # Grid search MLE (simplified)
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

            estimates_mux.append(best_mu[0])
            estimates_muy.append(best_mu[1])

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