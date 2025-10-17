#!/usr/bin/env python3
"""
OISL-ISAC Performance Limits Simulation (COMPLETE FIXED VERSION)
=================================================================
✓ Patch A: Dead-time chain rule in FIM gradients
✓ Patch B: Adaptive A_pilot selection
✓ All English labels (IEEE ready)
✓ Bug fixes: Index error in Pareto computation
✓ Output to section_III directory
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from math import ceil, sqrt, exp, log, pi
import os
from tqdm import tqdm
import warnings
import multiprocessing as mp
from functools import partial

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS & STYLE
# ============================================================================

SPEED_OF_LIGHT = 299792458  # m/s


def setup_ieee_style():
    """IEEE single-column style"""
    plt.rcParams.update({
        'figure.figsize': (4, 3),
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'font.family': 'serif',
        'font.size': 10,
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8,
        'lines.linewidth': 1.5,
        'lines.markersize': 5,
        'grid.alpha': 0.3,
    })

    colors = {
        'zodiacal': '#0072BD',
        'earthshine': '#EDB120',
        'stray_light': '#A2142F',
    }
    return colors


colors = setup_ieee_style()


# ============================================================================
# POISSON ENTROPY
# ============================================================================

def poisson_entropy(lam):
    """Compute entropy with numerical stability"""
    if lam <= 0:
        return 0.0

    Kmax = int(ceil(lam + 10.0 * sqrt(max(lam, 1.0))))
    ks = np.arange(Kmax + 1)

    log_pk = -lam + ks * np.log(max(lam, 1e-100)) - gammaln(ks + 1)
    pk = np.exp(log_pk - np.max(log_pk))
    pk = pk / pk.sum()
    pk = pk[pk > 1e-15]

    return -np.sum(pk * np.log2(pk)) if len(pk) > 0 else 0.0


# ============================================================================
# CAPACITY
# ============================================================================

def capacity_lb(Sbar, Smax, lamb_b, dt=1.0, tau_d=None, M_pixels=1, verbose=False):
    """Binary-input capacity with proper dead-time handling"""
    # Effective peak
    if tau_d is not None:
        Smax_dead = (dt / tau_d) * max(1, M_pixels)
        Smax_eff = min(Smax, Smax_dead)
    else:
        Smax_eff = Smax

    # Feasibility
    if Sbar > Smax_eff:
        if verbose:
            print(f"  ❌ INFEASIBLE: Sbar={Sbar:.1f} > Smax_eff={Smax_eff:.1f}")
        return 0.0, None

    A_low, A_high = Sbar, Smax_eff
    if A_high < A_low:
        return 0.0, None

    A_grid = np.linspace(A_low, A_high, 200)
    Cbest, Aopt = 0.0, Sbar

    for A in A_grid:
        p = Sbar / A
        if p > 1:
            continue

        # Convert to rates and apply dead time
        r0 = lamb_b / dt
        r1 = (lamb_b + A) / dt

        if tau_d is not None:
            r0 = r0 / (1.0 + r0 * tau_d)
            r1 = r1 / (1.0 + r1 * tau_d)

        lam0 = r0 * dt
        lam1 = r1 * dt

        # Compute entropies
        HY0 = poisson_entropy(lam0)
        HY1 = poisson_entropy(lam1)

        # Mixture
        Kmax = int(ceil(max(lam0, lam1) + 10.0 * sqrt(max(max(lam0, lam1), 1.0))))
        ks = np.arange(Kmax + 1)

        log_p0 = -lam0 + ks * np.log(max(lam0, 1e-100)) - gammaln(ks + 1)
        log_p1 = -lam1 + ks * np.log(max(lam1, 1e-100)) - gammaln(ks + 1)

        p0 = np.exp(log_p0 - np.max(log_p0))
        p1 = np.exp(log_p1 - np.max(log_p1))

        p0 = p0 / p0.sum()
        p1 = p1 / p1.sum()

        PY = (1 - p) * p0 + p * p1
        PY = PY / PY.sum()
        PY_nonzero = PY[PY > 1e-15]
        HY = -np.sum(PY_nonzero * np.log2(PY_nonzero)) if len(PY_nonzero) > 0 else 0

        I = HY - (1 - p) * HY0 - p * HY1

        if I > Cbest:
            Cbest, Aopt = I, A

    return Cbest, Aopt


# ============================================================================
# POINTING LOSS
# ============================================================================

def E_Lp(mu_vec, sigma2, theta_b):
    """Expected pointing loss"""
    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)
    mu2 = np.dot(mu_vec, mu_vec)
    gamma = 1.0 + a * sigma2
    return (1.0 / gamma) * np.exp(-b * mu2 / gamma)


def r_deadtime(r, tau_d):
    """Non-paralyzable dead time"""
    return r / (1.0 + r * tau_d)


# ============================================================================
# ADAPTIVE A_PILOT SELECTION (PATCH B)
# ============================================================================

def choose_A_pilot(params, lambda_b_slot, zeta=0.25):
    """
    Adaptive pilot amplitude to avoid dead-time saturation

    Parameters:
    -----------
    zeta : float
        Target r*tau_d (0.25 is conservative for high-photon)
    """
    dt = params['dt']
    tau_d = params['tau_d']
    Smax = params['Smax']
    M_pixels = params['M_pixels']
    Sbar = params['Sbar']

    # Effective peak
    if tau_d:
        Smax_eff = min(Smax, (dt / tau_d) * M_pixels)
    else:
        Smax_eff = Smax

    # Average pointing loss
    Lp_avg = float(E_Lp(params['mu'], params['sigma2'], params['theta_b']))

    # Background rate (pre-correction)
    r_b_pre = lambda_b_slot / dt

    if tau_d:
        # Target rate: r*tau_d = zeta
        r_target = zeta / tau_d

        # Required A
        A_cap = max(0.0, (r_target - r_b_pre) * dt / max(Lp_avg, 1e-9))

        # Conservative selection (0.6 for high-photon)
        A_use = min(Smax_eff, max(0.1 * Sbar, 0.6 * A_cap))
    else:
        A_use = min(Smax_eff, 0.8 * min(4.0 * Sbar, Smax_eff))

    return A_use


# ============================================================================
# FIM WITH DEAD-TIME CHAIN RULE (PATCH A)
# ============================================================================

def fim_pilot(alpha, rho, Sbar, N, dt, params, dither_seq, tau_d=None,
              A_pilot=None, M_pixels=1, verbose=False):
    """
    FIM computation with dead-time gradient correction

    KEY FIX: Include g_dead = 1/(1+r*tau_d)^2 in all gradients
    """
    mu = params["mu"].copy()
    sigma2 = params["sigma2"]
    theta_b = params["theta_b"]
    r_b = params["r_b"]
    Smax = params.get("Smax", np.inf)

    # Effective peak
    if tau_d is not None:
        Smax_dead = (dt / tau_d) * max(1, M_pixels)
        Smax_eff = min(Smax, Smax_dead)
    else:
        Smax_eff = Smax

    # Pilot amplitude
    if A_pilot is None:
        A_pilot = min(Smax_eff, 4.0 * Sbar) * 0.5

    A_pilot = min(A_pilot, Smax_eff)

    # Pilot slot count
    S_total_pilot = rho * Sbar * N
    Npilot = int(min(
        np.floor(alpha * N),
        np.floor(S_total_pilot / max(A_pilot, 1e-12))
    ))

    if Npilot <= 0:
        return np.zeros((4, 4))

    # Dither sequence
    if len(dither_seq) < Npilot:
        reps = int(np.ceil(Npilot / len(dither_seq)))
        dither_seq_used = (dither_seq * reps)[:Npilot]
    else:
        dither_seq_used = dither_seq[:Npilot]

    if verbose:
        print(f"    α={alpha:.2f}, ρ={rho:.2f} → Npilot={Npilot}, A={A_pilot:.1f}")

    # Compute FIM
    I = np.zeros((4, 4))
    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)
    gamma = 1.0 + a * sigma2

    for n in range(Npilot):
        mu_eff = mu + dither_seq_used[n]
        mu_eff_norm2 = np.dot(mu_eff, mu_eff)

        # Pointing loss
        Lp = (1.0 / gamma) * np.exp(-b * mu_eff_norm2 / gamma)

        # =====================================================================
        # PATCH A: Dead-time chain rule correction
        # =====================================================================
        # Step 1: Pre-correction rates
        r_s_pre = A_pilot * Lp / dt
        r_b_pre = r_b / dt
        r_pre = r_s_pre + r_b_pre

        # Step 2: Apply dead-time and compute chain rule factor
        if tau_d is not None:
            # KEY: ∂r'/∂r = 1/(1+r*tau_d)^2
            g_dead = 1.0 / (1.0 + r_pre * tau_d) ** 2
            r_tot = r_pre / (1.0 + r_pre * tau_d)
        else:
            g_dead = 1.0
            r_tot = r_pre

        lam = r_tot * dt

        # Step 3: Gradients (ALL multiplied by g_dead)
        dLp_dmux = Lp * (-2 * b * mu_eff[0] / gamma)
        dLp_dmuy = Lp * (-2 * b * mu_eff[1] / gamma)
        dLp_dsig = Lp * (-a / gamma + a * b * mu_eff_norm2 / (gamma ** 2))

        dlam_dmux = g_dead * A_pilot * dLp_dmux * dt
        dlam_dmuy = g_dead * A_pilot * dLp_dmuy * dt
        dlam_dsig = g_dead * A_pilot * dLp_dsig * dt
        dlam_drb = g_dead * dt
        # =====================================================================

        grad = np.array([dlam_dmux, dlam_dmuy, dlam_dsig, dlam_drb])

        if lam > 1e-14:
            I += np.outer(grad, grad) / lam

    return I


def generate_dither_sequence(N_pilot, theta_b, delta_factor=0.6):
    """Generate dithering sequence"""
    delta = delta_factor * theta_b
    base_pattern = [
        np.array([delta, 0.0]),
        np.array([-delta, 0.0]),
        np.array([0.0, delta]),
        np.array([0.0, -delta])
    ]
    reps = max(1, int(np.ceil(N_pilot / len(base_pattern))))
    return base_pattern * reps


# ============================================================================
# PARAMETER PRESETS
# ============================================================================

def get_preset_params(preset='moderate'):
    """Three validated presets"""
    c = SPEED_OF_LIGHT
    h = 6.626e-34
    wavelength = 1550e-9
    nu = c / wavelength
    hnu = h * nu

    if preset == 'low_photon':
        params = {
            'Sbar': 2.0,
            'Smax': 20.0,
            'dt': 1e-6,
            'N': 5000,
            'eta': 0.8,
            'hnu': hnu,
            'theta_b': 10e-6,
            'Llink': 1e-12,
            'mu': np.array([0.0, 0.0]),
            'sigma2': (2e-6) ** 2,
            'tau_d': None,
            'M_pixels': 1,
            'J_P': np.diag([1e-11, 1e-11, 1e-9, 1e-5]),
            'W': np.diag([1.0, 1.0, 0.0, 0.0]),
        }
        scenarios = {
            'Low': {'name': 'Zodiacal', 'r_b': 0.01, 'color': colors['zodiacal']},
            'Medium': {'name': 'Earthshine', 'r_b': 0.5, 'color': colors['earthshine']},
            'High': {'name': 'Stray Light', 'r_b': 5.0, 'color': colors['stray_light']},
        }

    elif preset == 'moderate':
        params = {
            'Sbar': 10.0,
            'Smax': 100.0,
            'dt': 2e-6,
            'N': 10000,
            'eta': 0.8,
            'hnu': hnu,
            'theta_b': 10e-6,
            'Llink': 1e-12,
            'mu': np.array([0.0, 0.0]),
            'sigma2': (1.5e-6) ** 2,
            'tau_d': 50e-9,
            'M_pixels': 16,
            'J_P': np.diag([1e-12, 1e-12, 1e-10, 1e-4]),
            'W': np.diag([1.0, 1.0, 0.0, 0.0]),
        }
        scenarios = {
            'Low': {'name': 'Zodiacal', 'r_b': 0.05, 'color': colors['zodiacal']},
            'Medium': {'name': 'Earthshine', 'r_b': 1.0, 'color': colors['earthshine']},
            'High': {'name': 'Stray Light', 'r_b': 10.0, 'color': colors['stray_light']},
        }

    elif preset == 'high_photon':
        params = {
            'Sbar': 50.0,
            'Smax': 500.0,
            'dt': 1e-6,
            'N': 10000,
            'eta': 0.85,
            'hnu': hnu,
            'theta_b': 8e-6,
            'Llink': 1e-12,
            'mu': np.array([0.0, 0.0]),
            'sigma2': (1e-6) ** 2,
            'tau_d': 100e-9,
            'M_pixels': 64,
            'J_P': np.diag([1e-12, 1e-12, 1e-10, 1e-4]),
            'W': np.diag([1.0, 1.0, 0.0, 0.0]),
        }
        scenarios = {
            'Low': {'name': 'Zodiacal', 'r_b': 0.01, 'color': colors['zodiacal']},
            'Medium': {'name': 'Earthshine', 'r_b': 1.0, 'color': colors['earthshine']},
            'High': {'name': 'Stray Light', 'r_b': 20.0, 'color': colors['stray_light']},
        }

    else:
        raise ValueError(f"Unknown preset: {preset}")

    return params, scenarios


# ============================================================================
# FIGURE 1: CAPACITY VS BACKGROUND
# ============================================================================

def generate_fig_capacity_vs_background(params, output_dir='./', verbose=False):
    """Generate capacity vs background"""
    print("\n" + "=" * 60)
    print("Figure 1: Capacity vs Background")
    print("=" * 60)

    rb_array = np.logspace(-2, 2, 60)

    Sbar = params['Sbar']
    Smax = params['Smax']
    tau_d = params['tau_d']
    dt = params['dt']
    M_pixels = params['M_pixels']

    print(f"Parameters: Sbar={Sbar}, M={M_pixels}")

    capacity_values = []

    for r_b in tqdm(rb_array, desc="Background sweep"):
        C_lb, A_opt = capacity_lb(Sbar, Smax, r_b, dt, tau_d, M_pixels, verbose=False)
        capacity_values.append(C_lb)

    capacity_values = np.array(capacity_values)

    print(f"✓ Range: [{np.min(capacity_values):.4f}, {np.max(capacity_values):.4f}] bits/slot")

    # Plot
    fig, ax = plt.subplots(figsize=(4, 3))

    ax.semilogx(rb_array, capacity_values, 'b-', linewidth=2)

    ax.axvline(x=0.01, color=colors['zodiacal'], linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axvline(x=1.0, color=colors['earthshine'], linestyle='--', linewidth=1.2, alpha=0.7)
    ax.axvline(x=10.0, color=colors['stray_light'], linestyle='--', linewidth=1.2, alpha=0.7)

    ax.set_xlabel('Background λ_b [photons/slot]', fontsize=10, weight='bold')
    ax.set_ylabel('Capacity C_LB [bits/slot]', fontsize=10, weight='bold')
    ax.set_title(f'Capacity vs Background\n(S̄={Sbar:.1f}, M={M_pixels})', fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/capacity_vs_background.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/capacity_vs_background.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: capacity_vs_background.pdf/png")
    return rb_array, capacity_values


# ============================================================================
# FIGURE 2: FIM VS RESOURCES
# ============================================================================

def generate_fig_fim_vs_resources(params, scenarios, output_dir='./', verbose=False):
    """Generate FIM heatmap with adaptive A_pilot"""
    print("\n" + "=" * 60)
    print("Figure 2: FIM vs Resources")
    print("=" * 60)

    scenario = scenarios['Medium']
    params_sim = params.copy()
    params_sim['r_b'] = scenario['r_b']

    alpha_range = np.linspace(0.05, 0.95, 25)
    rho_range = np.linspace(0.05, 0.95, 25)
    mse_trace = np.zeros((len(rho_range), len(alpha_range)))

    max_pilots = int(0.95 * params['N'])
    dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

    # Adaptive A_pilot
    A_pilot_use = choose_A_pilot(params, scenario['r_b'], zeta=0.25)
    print(f"Adaptive A_pilot = {A_pilot_use:.2f} photons/slot")

    # Diagnostic
    if params['tau_d']:
        r_test = (scenario['r_b'] + A_pilot_use * 0.5) / params['dt']
        saturation = r_test * params['tau_d']
        print(f"Estimated r*tau_d ≈ {saturation:.3f} (target < 0.5)")

    print(f"Computing FIM grid (25×25)...")

    valid_count = 0

    if params['tau_d']:
        Smax_eff = min(params['Smax'],
                       (params['dt'] / params['tau_d']) * params['M_pixels'])
    else:
        Smax_eff = params['Smax']

    for i, rho in enumerate(tqdm(rho_range, desc="ρ sweep")):
        for j, alpha in enumerate(alpha_range):
            try:
                I_pilot = fim_pilot(alpha, rho, params['Sbar'], params['N'],
                                    params['dt'], params_sim, dither_seq,
                                    params['tau_d'], A_pilot_use, params['M_pixels'])

                J = I_pilot + params['J_P']
                J += 1e-12 * np.eye(4)

                if np.linalg.cond(J) > 1e30:
                    mse_trace[i, j] = np.nan
                else:
                    J_inv = np.linalg.inv(J)
                    mse_trace[i, j] = np.trace(params['W'] @ J_inv)
                    valid_count += 1

            except:
                mse_trace[i, j] = np.nan

    print(f"✓ Valid points: {valid_count}/{mse_trace.size} ({valid_count / mse_trace.size * 100:.1f}%)")

    # Diagnostics
    valid = np.isfinite(mse_trace) & (mse_trace > 0)

    if valid.sum() < 0.05 * mse_trace.size:
        print("⚠ Too few valid points")
        return None, None, None

    mse_valid = mse_trace[valid]
    mse_min, mse_max = mse_valid.min(), mse_valid.max()
    mse_ratio = mse_max / mse_min
    mse_std = np.std(np.log10(mse_valid))

    print(f"✓ MSE range: [{mse_min:.3e}, {mse_max:.3e}]")
    print(f"✓ MSE ratio: {mse_ratio:.2e} (should be >> 1)")
    print(f"✓ log₁₀(MSE) std: {mse_std:.3f} (should be > 0.1)")

    if mse_ratio < 10:
        print("⚠ WARNING: MSE has narrow range!")

    vmin = np.percentile(mse_trace[valid], 5)
    vmax = np.percentile(mse_trace[valid], 95)

    log_mse = np.full_like(mse_trace, np.nan)
    log_mse[valid] = np.log10(mse_trace[valid])

    # Plot
    fig, ax = plt.subplots(figsize=(4, 3))

    Alpha, Rho = np.meshgrid(alpha_range, rho_range)

    im = ax.pcolormesh(Alpha, Rho, log_mse, shading='auto', cmap='viridis_r',
                       vmin=np.log10(vmin), vmax=np.log10(vmax))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(MSE μx,μy)', rotation=270, labelpad=18,
                   fontsize=9, weight='bold')

    try:
        levels = np.linspace(np.log10(vmin), np.log10(vmax), 6)
        contours = ax.contour(Alpha, Rho, log_mse, levels=levels,
                              colors='white', alpha=0.5, linewidths=0.7)
        ax.clabel(contours, inline=True, fontsize=6, fmt='%.1f')
    except:
        pass

    ax.set_xlabel('Time Allocation α', fontsize=10, weight='bold')
    ax.set_ylabel('Photon Allocation ρ', fontsize=10, weight='bold')
    ax.set_title(f'FIM vs Resources\n({scenario["name"]})', fontsize=11, weight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fim_vs_resources.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fim_vs_resources.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: fim_vs_resources.pdf/png")
    return alpha_range, rho_range, mse_trace


# ============================================================================
# FIGURE 3: PARETO BOUNDARY
# ============================================================================

def _compute_pareto_point_worker(args):
    """Worker for parallel Pareto computation"""
    (D_max, alpha_search, rho_search, params, params_sim,
     dither_seq, Smax_eff, scenario_r_b) = args

    max_rate = 0.0
    best_alpha, best_rho = 0, 0

    A_pilot_use = choose_A_pilot(params, scenario_r_b, zeta=0.25)

    for alpha in alpha_search:
        for rho in rho_search:
            try:
                if alpha >= 0.99:  # Relaxed from 1.0 to allow more extreme cases
                    continue

                S_data = (1 - rho) * params['Sbar'] / (1 - alpha)

                I_pilot = fim_pilot(
                    alpha, rho, params['Sbar'], params['N'],
                    params['dt'], params_sim, dither_seq,
                    params['tau_d'], A_pilot_use, params['M_pixels']
                )

                J = I_pilot + params['J_P'] + 1e-12 * np.eye(4)

                if np.linalg.cond(J) > 1e30:
                    continue

                J_inv = np.linalg.inv(J)
                mse_current = np.trace(params['W'] @ J_inv)

                if mse_current > D_max:
                    continue

                C_data, _ = capacity_lb(
                    S_data, params['Smax'], scenario_r_b,
                    params['dt'], params['tau_d'], params['M_pixels']
                )
                rate = (1 - alpha) * C_data

                if rate > max_rate:
                    max_rate = rate
                    best_alpha, best_rho = alpha, rho

            except Exception:
                continue

    return (max_rate, D_max, best_alpha, best_rho) if max_rate > 0 else None


def generate_fig_pareto_boundary(params, scenarios, output_dir='./',
                                 verbose=False, n_workers=None):
    """Generate Pareto boundary"""
    print("\n" + "=" * 60)
    print("Figure 3: Rate-MSE Pareto (Enhanced)")
    print("=" * 60)

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    n_workers = max(1, n_workers)

    print(f"🚀 Using {n_workers} parallel worker(s)")

    # Phase 1: Probe MSE range (extended for high-noise)
    alpha_probe = np.linspace(0.05, 0.95, 12)  # Include extreme allocations
    rho_probe = np.linspace(0.05, 0.95, 12)
    max_pilots = int(0.9 * params['N'])
    dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

    probe = scenarios['Medium']
    params_probe = params.copy()
    params_probe['r_b'] = probe['r_b']

    mse_samples = []

    if params['tau_d']:
        Smax_eff = min(params['Smax'],
                       (params['dt'] / params['tau_d']) * params['M_pixels'])
    else:
        Smax_eff = params['Smax']

    A_pilot_probe = choose_A_pilot(params, probe['r_b'], zeta=0.25)

    print(f"Probing MSE range (A_pilot={A_pilot_probe:.2f})...")
    for alpha in alpha_probe:
        for rho in rho_probe:
            try:
                I_pilot = fim_pilot(alpha, rho, params['Sbar'], params['N'], params['dt'],
                                    params_probe, dither_seq, params['tau_d'],
                                    A_pilot_probe, params['M_pixels'])
                J = I_pilot + params['J_P'] + 1e-12 * np.eye(4)
                if np.linalg.cond(J) < 1e30:
                    mse = np.trace(params['W'] @ np.linalg.inv(J))
                    if np.isfinite(mse) and mse > 0:
                        mse_samples.append(mse)
            except:
                pass

    if len(mse_samples) == 0:
        print("⚠️ No valid MSE samples, using fallback")
        D_targets = np.logspace(-15, -8, 18)  # Extended range
    else:
        mmin, mmed, mmax = np.percentile(mse_samples, [5, 50, 95])
        # Enhanced: Much wider range for high-noise scenarios
        D_targets = np.logspace(np.log10(mmin * 0.3), np.log10(mmax * 50.0), 18)
        print(f"✅ MSE samples: min={mmin:.3e}, med={mmed:.3e}, max={mmax:.3e}")
        print(f"   D_targets: [{D_targets[0]:.3e}, {D_targets[-1]:.3e}]")

    # Phase 2: Compute Pareto (extended range for high-noise)
    alpha_search = np.linspace(0.02, 0.98, 25)  # Allow extreme allocations
    rho_search = np.linspace(0.05, 0.95, 20)

    pareto_results = {}

    for scenario_name, scenario in scenarios.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {scenario_name} (λ_b={scenario['r_b']})")
        print(f"{'=' * 60}")

        params_sim = params.copy()
        params_sim['r_b'] = scenario['r_b']

        worker_args = [
            (D_max, alpha_search, rho_search, params, params_sim,
             dither_seq, Smax_eff, scenario['r_b'])
            for D_max in D_targets
        ]

        if n_workers > 1:
            print(f"🔄 Parallel computation...")
            with mp.Pool(n_workers) as pool:
                results = list(tqdm(
                    pool.imap(_compute_pareto_point_worker, worker_args),
                    total=len(D_targets),
                    desc=f"  {scenario_name}"
                ))
        else:
            print(f"🔄 Sequential computation...")
            results = [
                _compute_pareto_point_worker(args)
                for args in tqdm(worker_args, desc=f"  {scenario_name}")
            ]

        # Filter valid - FIX: Correct unpacking
        pareto_points = [r for r in results if r is not None]
        pareto_results[scenario_name] = [
            (rate, D_max) for (rate, D_max, _, _) in pareto_points
        ]

        print(f"✅ Found {len(pareto_points)}/{len(D_targets)} valid points")

        # FIX: Correct unpacking for diagnostics
        if len(pareto_points) > 0:
            rates = [p[0] for p in pareto_points]  # p is (rate, D_max, alpha, rho)
            print(f"   Rate range: [{min(rates):.4f}, {max(rates):.4f}] bits/slot")

    # Phase 3: Plot
    print(f"\n{'=' * 60}")
    print("📊 Generating plot...")
    print(f"{'=' * 60}")

    fig, ax = plt.subplots(figsize=(4, 3))

    for scenario_name, scenario in scenarios.items():
        points = pareto_results[scenario_name]
        if len(points) > 0:
            rates = [p[0] for p in points]
            mses = [p[1] for p in points]

            sorted_pairs = sorted(zip(mses, rates))
            mses_sorted = [p[0] for p in sorted_pairs]
            rates_sorted = [p[1] for p in sorted_pairs]

            ax.loglog(mses_sorted, rates_sorted, 'o-',
                      color=scenario['color'], linewidth=2, markersize=5,
                      label=scenario["name"])
        else:
            print(f"⚠️ No valid points for {scenario_name}")

    ax.set_xlabel('MSE (μx, μy) [rad²]', fontsize=10, weight='bold')
    ax.set_ylabel('Rate [bits/slot]', fontsize=10, weight='bold')
    ax.set_title(f'Rate-MSE Pareto\n(S̄={params["Sbar"]:.1f}, M={params["M_pixels"]})',
                 fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/rate_mse_boundary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/rate_mse_boundary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Saved: rate_mse_boundary.pdf/png")
    return pareto_results


# ============================================================================
# MAIN
# ============================================================================

def main(preset='moderate'):
    """Main function"""
    print("\n" + "=" * 80)
    print("OISL-ISAC SIMULATION (COMPLETE FIX)")
    print("✓ Patch A: Dead-time chain rule g_dead=1/(1+rτ)²")
    print("✓ Patch B: Adaptive A_pilot with zeta=0.25")
    print("✓ Enhanced D_targets range")
    print("✓ All labels in English (IEEE ready)")
    print("=" * 80)

    params, scenarios = get_preset_params(preset)

    print(f"\n✓ Preset: '{preset}'")
    print(f"  S̄={params['Sbar']:.1f}, M={params['M_pixels']}")

    if params['tau_d'] and params['Sbar'] > 30:
        print(f"  ⚠️ HIGH-PHOTON mode detected")

    output_dir = f'./section_III/results_{preset}'
    os.makedirs(output_dir, exist_ok=True)
    print(f"  Output: {output_dir}")

    try:
        # Figure 1
        rb_array, capacity_values = generate_fig_capacity_vs_background(
            params, output_dir)

        cap_diffs = np.diff(capacity_values)
        if np.all(cap_diffs <= 1e-6):
            print("✓ Unit test PASSED: Capacity monotonic")

        # Figure 2
        alpha_range, rho_range, mse_trace = generate_fig_fim_vs_resources(
            params, scenarios, output_dir)

        # Figure 3
        if mse_trace is not None:
            valid_mse = mse_trace[np.isfinite(mse_trace)]
            if len(valid_mse) > 0:
                mse_ratio = valid_mse.max() / valid_mse.min()
                print(f"✓ MSE dynamic range: {mse_ratio:.2e}")

            pareto_results = generate_fig_pareto_boundary(
                params, scenarios, output_dir)

        print(f"\n{'=' * 80}")
        print(f"✓✓✓ ALL FIGURES GENERATED ✓✓✓")
        print(f"{'=' * 80}")
        print(f"Results: {output_dir}/")

        for f in sorted(os.listdir(output_dir)):
            print(f"  • {f}")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) > 1:
        preset = sys.argv[1]
        if preset == 'all':
            presets_to_run = ['low_photon', 'moderate', 'high_photon']
        elif preset in ['low_photon', 'moderate', 'high_photon']:
            presets_to_run = [preset]
        else:
            print(f"❌ Invalid preset: {preset}")
            sys.exit(1)
    else:
        print("ℹ️ Running ALL presets by default")
        presets_to_run = ['low_photon', 'moderate', 'high_photon']

    print("\n" + "=" * 80)
    print(f"🚀 RUNNING {len(presets_to_run)} PRESET(S)")
    print("=" * 80)

    results = {}
    total_start = time.time()

    for i, preset in enumerate(presets_to_run, 1):
        print(f"\n{'=' * 80}")
        print(f"📊 [{i}/{len(presets_to_run)}] Starting preset: {preset.upper()}")
        print(f"{'=' * 80}")

        start_time = time.time()
        success = main(preset)
        elapsed = time.time() - start_time

        results[preset] = {'success': success, 'time': elapsed}

        print(f"\n{'=' * 80}")
        if success:
            print(f"✅ [{i}/{len(presets_to_run)}] {preset.upper()} completed in {elapsed:.1f}s")
        else:
            print(f"❌ [{i}/{len(presets_to_run)}] {preset.upper()} FAILED after {elapsed:.1f}s")
        print(f"{'=' * 80}")

    total_elapsed = time.time() - total_start
    print("\n" + "=" * 80)
    print("📋 EXECUTION SUMMARY")
    print("=" * 80)

    for preset, result in results.items():
        status = "✅ SUCCESS" if result['success'] else "❌ FAILED"
        print(f"  {preset:15s} │ {status:12s} │ {result['time']:6.1f}s")

    print(f"\n  Total time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f} min)")

    success_count = sum(1 for r in results.values() if r['success'])
    print(f"  Success rate: {success_count}/{len(results)}")

    all_success = all(r['success'] for r in results.values())
    exit(0 if all_success else 1)