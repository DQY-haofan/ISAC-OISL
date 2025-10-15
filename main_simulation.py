#!/usr/bin/env python3
"""
OISL-ISAC Performance Limits Simulation (FIXED VERSION)
========================================================

All critical patches applied:
✓ Patch 1: Multi-pixel SPAD array + feasibility checks
✓ Patch 2: Fixed α-ρ decoupling in FIM computation
✓ Patch 3: Robust heatmap visualization
✓ Patch 4: Numerical stability with gammaln

Three parameter presets included for immediate visualization.

Author: Fixed by Expert Analysis
Date: 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln  # Patch 4: Numerical stability
from math import ceil, sqrt, exp, log, pi
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS & STYLE
# ============================================================================

SPEED_OF_LIGHT = 299792458  # m/s


def setup_ieee_style():
    """IEEE single-column style with large fonts"""
    plt.rcParams.update({
        'figure.figsize': (3.5, 2.8),
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
        'low_M': '#77AC30',
        'high_M': '#7E2F8E',
    }
    return colors


colors = setup_ieee_style()


# ============================================================================
# PATCH 4: IMPROVED POISSON ENTROPY (NUMERICAL STABILITY)
# ============================================================================

def poisson_entropy(lam):
    """
    Compute entropy of Poisson distribution with numerical stability.
    Uses gammaln instead of loop for log(k!)
    """
    if lam <= 0:
        return 0.0

    Kmax = int(ceil(lam + 10.0 * sqrt(max(lam, 1.0))))
    ks = np.arange(Kmax + 1)

    # Numerically stable: log(k!) = gammaln(k+1)
    log_pk = -lam + ks * np.log(max(lam, 1e-100)) - gammaln(ks + 1)
    pk = np.exp(log_pk - np.max(log_pk))  # Subtract max for stability
    pk = pk / pk.sum()
    pk = pk[pk > 1e-15]

    return -np.sum(pk * np.log2(pk)) if len(pk) > 0 else 0.0


# ============================================================================
# PATCH 1: CAPACITY WITH MULTI-PIXEL FEASIBILITY
# ============================================================================

def capacity_lb(Sbar, Smax, lamb_b, dt=1.0, tau_d=None, M_pixels=1, verbose=False):
    """
    Binary-input capacity lower bound with feasibility guard.

    Parameters:
    -----------
    Sbar : float
        Average signal photons per slot
    Smax : float
        Peak photons per slot (hardware limit)
    lamb_b : float
        Background photons per slot
    dt : float
        Time slot duration [s]
    tau_d : float or None
        Dead time [s]
    M_pixels : int
        Number of parallel SPAD pixels (default: 1)
    verbose : bool
        Print diagnostic info

    Returns:
    --------
    C_lb : float
        Capacity lower bound [bits/slot]
    A_opt : float or None
        Optimal signal amplitude
    """
    # Effective peak under dead-time + pixel parallelism
    if tau_d is not None:
        Smax_dead = (dt / tau_d) * max(1, M_pixels)
        Smax_eff = min(Smax, Smax_dead)
    else:
        Smax_eff = Smax

    # Feasibility check: need A >= Sbar to have p<=1
    if Sbar > Smax_eff:
        if verbose:
            print(f"  ❌ INFEASIBLE: Sbar={Sbar:.1f} > Smax_eff={Smax_eff:.1f}")
            print(f"     Solution: Increase dt, reduce tau_d, or use M_pixels > {M_pixels}")
        return 0.0, None

    A_low, A_high = Sbar, Smax_eff
    if A_high < A_low:
        return 0.0, None

    # Optimize over feasible range
    A_grid = np.linspace(A_low, A_high, 200)
    Cbest, Aopt = 0.0, Sbar

    for A in A_grid:
        p = Sbar / A
        if p > 1:
            continue

        lam0 = lamb_b
        lam1 = lamb_b + A

        # Apply dead time if specified
        if tau_d is not None:
            lam0 = lam0 / (1.0 + (lam0 / dt) * tau_d) if lam0 / dt * tau_d < 1 else 0
            lam1 = lam1 / (1.0 + (lam1 / dt) * tau_d) if lam1 / dt * tau_d < 1 else 0

        # Compute entropies
        HY0 = poisson_entropy(lam0)
        HY1 = poisson_entropy(lam1)

        # Mixture distribution
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

        # Mutual information
        I = HY - (1 - p) * HY0 - p * HY1

        if I > Cbest:
            Cbest, Aopt = I, A

    if verbose:
        print(f"  ✓ FEASIBLE: C_lb={Cbest:.4f} bits/slot at A={Aopt:.1f}")

    return Cbest, Aopt


# ============================================================================
# PATCH 2: FIM WITH FIXED A_pilot AND DITHER TILING
# ============================================================================

def E_Lp(mu_vec, sigma2, theta_b):
    """Expected pointing loss"""
    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)
    mu2 = np.dot(mu_vec, mu_vec)
    gamma = 1.0 + a * sigma2
    return (1.0 / gamma) * np.exp(-b * mu2 / gamma)


def r_deadtime(r, tau_d):
    """Non-paralyzable dead time correction"""
    return r / (1.0 + r * tau_d)


def fim_pilot(alpha, rho, Sbar, N, dt, params, dither_seq, tau_d=None,
              A_pilot=None, M_pixels=1, verbose=False):
    """
    Fisher Information Matrix computation with α-ρ decoupling.

    Key fix: A_pilot is fixed per-slot amplitude, α controls number of slots.
    This breaks the Npilot * S_pilot cancellation.

    Parameters:
    -----------
    A_pilot : float or None
        Photons per pilot slot (fixed). If None, auto-set to 0.5*Smax_eff
    M_pixels : int
        Number of parallel SPAD pixels
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

    # Choose per-slot pilot amplitude
    if A_pilot is None:
        A_pilot = min(Smax_eff, 4.0 * Sbar) * 0.5  # Conservative default

    # Total pilot photons available in frame
    S_total_pilot = rho * Sbar * N

    # Number of pilot slots we can afford
    Npilot = int(min(
        np.floor(alpha * N),
        np.floor(S_total_pilot / max(A_pilot, 1e-12))
    ))

    if Npilot <= 0:
        return np.zeros((4, 4))

    # Ensure dither covers Npilot (tile if needed)
    if len(dither_seq) < Npilot:
        reps = int(np.ceil(Npilot / len(dither_seq)))
        dither_seq_used = (dither_seq * reps)[:Npilot]
    else:
        dither_seq_used = dither_seq[:Npilot]

    if verbose:
        print(f"    FIM: α={alpha:.2f}, ρ={rho:.2f} → Npilot={Npilot}, A_pilot={A_pilot:.1f}")

    # Compute FIM
    I = np.zeros((4, 4))
    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)
    gamma = 1.0 + a * sigma2

    for n in range(Npilot):
        mu_eff = mu + dither_seq_used[n]
        Lp = float(E_Lp(mu_eff, sigma2, theta_b))

        # Per-slot rate with dead-time correction
        r_s = A_pilot * Lp / dt
        r_tot = r_s + r_b

        if tau_d is not None and r_tot * tau_d > 1e-3:
            r_tot = r_deadtime(r_tot, tau_d)

        lam = r_tot * dt

        # Partial derivatives
        dlam_dmux = A_pilot * Lp * (-2 * b * mu_eff[0] / gamma)
        dlam_dmuy = A_pilot * Lp * (-2 * b * mu_eff[1] / gamma)
        dlam_dsig = A_pilot * Lp * (-a / gamma + a * b * np.dot(mu_eff, mu_eff) / (gamma ** 2))
        dlam_drb = dt

        grad = np.array([dlam_dmux, dlam_dmuy, dlam_dsig, dlam_drb])

        if lam > 1e-14:
            I += np.outer(grad, grad) / lam

    return I


def generate_dither_sequence(N_pilot, theta_b, delta_factor=0.6):
    """Generate dithering sequence (will be tiled if needed)"""
    delta = delta_factor * theta_b
    base_pattern = [
        np.array([delta, 0.0]),
        np.array([-delta, 0.0]),
        np.array([0.0, delta]),
        np.array([0.0, -delta])
    ]
    # Create long enough sequence
    reps = max(1, int(np.ceil(N_pilot / len(base_pattern))))
    return base_pattern * reps


# ============================================================================
# PARAMETER PRESETS (3 SCENARIOS)
# ============================================================================

def get_preset_params(preset='low_photon'):
    """
    Three validated parameter presets for different physics demonstrations.

    Presets:
    --------
    'low_photon' : HPE regime, strong background effect
    'moderate'   : Balanced, good dynamic range
    'high_photon': Array parallelism showcase
    """
    c = SPEED_OF_LIGHT
    h = 6.626e-34
    wavelength = 1550e-9
    nu = c / wavelength
    hnu = h * nu

    if preset == 'low_photon':
        # Low photon budget: Shows strong capacity degradation with background
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
            'tau_d': None,  # No dead time for simplicity
            'M_pixels': 1,
            'J_P': np.diag([1e-11, 1e-11, 1e-9, 1e-5]),
            'W': np.eye(4),
        }
        scenarios = {
            'Low': {'name': 'Zodiacal', 'r_b': 0.01, 'color': colors['zodiacal']},
            'Medium': {'name': 'Earthshine', 'r_b': 0.5, 'color': colors['earthshine']},
            'High': {'name': 'Stray Light', 'r_b': 5.0, 'color': colors['stray_light']},
        }

    elif preset == 'moderate':
        # Moderate regime: Good for all three figures
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
            'M_pixels': 16,  # Small array
            'J_P': np.diag([1e-12, 1e-12, 1e-10, 1e-6]),
            'W': np.eye(4),
        }
        scenarios = {
            'Low': {'name': 'Zodiacal', 'r_b': 0.05, 'color': colors['zodiacal']},
            'Medium': {'name': 'Earthshine', 'r_b': 1.0, 'color': colors['earthshine']},
            'High': {'name': 'Stray Light', 'r_b': 10.0, 'color': colors['stray_light']},
        }

    elif preset == 'high_photon':
        # High photon: Showcases array parallelism benefit
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
            'M_pixels': 64,  # Realistic SPAD array
            'J_P': np.diag([1e-12, 1e-12, 1e-10, 1e-6]),
            'W': np.eye(4),
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
# FIGURE 1: CAPACITY VS BACKGROUND (FIXED)
# ============================================================================

def generate_fig_capacity_vs_background(params, output_dir='./', verbose=False):
    """Generate capacity vs background with proper feasibility handling"""
    print("\n" + "=" * 60)
    print("Figure 1: Capacity vs Background Noise (FIXED)")
    print("=" * 60)

    # Extended background range for better dynamics
    rb_array = np.logspace(-2, 2, 60)

    Sbar = params['Sbar']
    Smax = params['Smax']
    tau_d = params['tau_d']
    dt = params['dt']
    M_pixels = params['M_pixels']

    print(f"\nParameters:")
    print(f"  Sbar = {Sbar}")
    print(f"  Smax = {Smax}")
    print(f"  dt = {dt * 1e6:.1f} μs")
    print(f"  tau_d = {tau_d * 1e9:.1f} ns" if tau_d else "  tau_d = None")
    print(f"  M_pixels = {M_pixels}")

    # Check feasibility
    if tau_d:
        Smax_eff = min(Smax, (dt / tau_d) * M_pixels)
        print(f"  Smax_eff = {Smax_eff:.1f} (dead-time limited)")
    else:
        Smax_eff = Smax
        print(f"  Smax_eff = {Smax_eff:.1f} (no dead-time)")

    if Sbar > Smax_eff:
        print(f"  ⚠ WARNING: Sbar > Smax_eff, will return zeros!")

    capacity_values = []
    A_opt_values = []

    print(f"\nComputing capacity...")
    for r_b in tqdm(rb_array, desc="Background sweep"):
        C_lb, A_opt = capacity_lb(Sbar, Smax, r_b, dt, tau_d, M_pixels,
                                  verbose=(verbose and r_b in [0.01, 1.0, 10.0]))
        capacity_values.append(C_lb)
        A_opt_values.append(A_opt if A_opt else 0)

    capacity_values = np.array(capacity_values)

    print(f"\n✓ Results:")
    print(f"  Capacity range: [{np.min(capacity_values):.4f}, {np.max(capacity_values):.4f}] bits/slot")
    print(f"  Non-zero points: {np.sum(capacity_values > 0)}/{len(capacity_values)}")

    # Plot
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.semilogx(rb_array, capacity_values, 'b-', linewidth=2,
                label=f'M={M_pixels} pixels')

    # Regime markers
    ax.axvline(x=0.01, color=colors['zodiacal'], linestyle='--',
               linewidth=1.2, alpha=0.7)
    ax.axvline(x=1.0, color=colors['earthshine'], linestyle='--',
               linewidth=1.2, alpha=0.7)
    ax.axvline(x=10.0, color=colors['stray_light'], linestyle='--',
               linewidth=1.2, alpha=0.7)

    ax.set_xlabel('Background λ_b [photons/slot]', fontsize=10, weight='bold')
    ax.set_ylabel('Capacity C_LB [bits/slot]', fontsize=10, weight='bold')
    ax.set_title(f'Capacity vs Background\n(S̄={Sbar:.1f}, M={M_pixels})',
                 fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)

    # Add regime labels
    ymax = np.max(capacity_values) if np.max(capacity_values) > 0 else 1
    ax.text(0.005, ymax * 0.85, 'Zodiacal', ha='center', fontsize=8, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(0.3, ymax * 0.65, 'Earthshine', ha='center', fontsize=8, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(30, ymax * 0.45, 'Stray', ha='center', fontsize=8, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/capacity_vs_background.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/capacity_vs_background.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: capacity_vs_background.pdf/png")
    return rb_array, capacity_values


# ============================================================================
# FIGURE 2: FIM VS RESOURCE ALLOCATION (PATCH 3: ROBUST PLOTTING)
# ============================================================================

def generate_fig_fim_vs_resources(params, scenarios, output_dir='./', verbose=False):
    """Generate FIM heatmap with robust visualization"""
    print("\n" + "=" * 60)
    print("Figure 2: FIM vs Resource Allocation (FIXED)")
    print("=" * 60)

    scenario = scenarios['Medium']
    params_sim = params.copy()
    params_sim['r_b'] = scenario['r_b']

    # Resource grids
    alpha_range = np.linspace(0.05, 0.95, 25)
    rho_range = np.linspace(0.05, 0.95, 25)
    mse_trace = np.zeros((len(rho_range), len(alpha_range)))

    # Generate dither
    max_pilots = int(0.95 * params['N'])
    dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

    print(f"\nComputing FIM grid (25×25 = 625 points)...")
    print(f"  Using fixed A_pilot strategy for α-ρ decoupling")

    valid_count = 0

    for i, rho in enumerate(tqdm(rho_range, desc="ρ sweep")):
        for j, alpha in enumerate(alpha_range):
            try:
                # Check peak constraints
                if params['tau_d']:
                    Smax_eff = min(params['Smax'],
                                   (params['dt'] / params['tau_d']) * params['M_pixels'])
                else:
                    Smax_eff = params['Smax']

                A_pilot_use = min(Smax_eff, 4.0 * params['Sbar']) * 0.5

                S_pilot = rho * params['Sbar'] / alpha
                S_data = (1 - rho) * params['Sbar'] / (1 - alpha)

                # Feasibility
                if S_pilot > Smax_eff or S_data > Smax_eff:
                    mse_trace[i, j] = np.nan
                    continue

                # Compute FIM
                I_pilot = fim_pilot(alpha, rho, params['Sbar'], params['N'],
                                    params['dt'], params_sim, dither_seq,
                                    params['tau_d'], A_pilot_use, params['M_pixels'],
                                    verbose=(verbose and i % 10 == 0 and j % 10 == 0))

                J = I_pilot + params['J_P']

                # Relaxed condition number threshold (Patch 3)
                if np.linalg.cond(J) > 1e14:
                    mse_trace[i, j] = np.nan
                else:
                    J_inv = np.linalg.inv(J)
                    mse_trace[i, j] = np.trace(params['W'] @ J_inv)
                    valid_count += 1

            except:
                mse_trace[i, j] = np.nan

    print(f"\n✓ Valid points: {valid_count}/{mse_trace.size} ({valid_count / mse_trace.size * 100:.1f}%)")

    # PATCH 3: Robust plotting with proper masking
    valid = np.isfinite(mse_trace) & (mse_trace > 0)

    if valid.sum() < 0.1 * mse_trace.size:
        print("⚠ Too few valid points, consider adjusting parameters")
        return None, None, None

    # Percentile-based colormap limits
    vmin = np.percentile(mse_trace[valid], 5)
    vmax = np.percentile(mse_trace[valid], 95)

    log_mse = np.full_like(mse_trace, np.nan)
    log_mse[valid] = np.log10(mse_trace[valid])

    # Plot
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    Alpha, Rho = np.meshgrid(alpha_range, rho_range)

    im = ax.pcolormesh(Alpha, Rho, log_mse, shading='auto', cmap='viridis_r',
                       vmin=np.log10(vmin), vmax=np.log10(vmax))

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(Trace CRLB)', rotation=270, labelpad=18,
                   fontsize=9, weight='bold')
    cbar.ax.tick_params(labelsize=8)

    # Contours
    try:
        levels = np.linspace(np.log10(vmin), np.log10(vmax), 8)
        contours = ax.contour(Alpha, Rho, log_mse, levels=levels,
                              colors='white', alpha=0.5, linewidths=0.7)
        ax.clabel(contours, inline=True, fontsize=6, fmt='%.1f')
    except:
        pass

    ax.set_xlabel('Time Allocation α', fontsize=10, weight='bold')
    ax.set_ylabel('Photon Allocation ρ', fontsize=10, weight='bold')
    ax.set_title(f'Fisher Info vs Resources\n({scenario["name"]}, λ_b={scenario["r_b"]})',
                 fontsize=11, weight='bold')

    # Annotations
    ax.text(0.05, 0.95, 'High\nMSE', transform=ax.transAxes, fontsize=8,
            weight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.text(0.75, 0.05, 'Low\nMSE', transform=ax.transAxes, fontsize=8,
            weight='bold', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fim_vs_resources.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fim_vs_resources.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: fim_vs_resources.pdf/png")
    return alpha_range, rho_range, mse_trace


# ============================================================================
# FIGURE 3: RATE-MSE PARETO BOUNDARY (FIXED)
# ============================================================================

def generate_fig_pareto_boundary(params, scenarios, output_dir='./', verbose=False):
    """Generate Pareto boundary with proper feasibility"""
    print("\n" + "=" * 60)
    print("Figure 3: Rate-MSE Pareto Boundary (FIXED)")
    print("=" * 60)

    # Coarse grids for speed
    D_targets = np.logspace(-8, -4, 15)
    alpha_search = np.linspace(0.1, 0.9, 10)
    rho_search = np.linspace(0.1, 0.9, 10)

    max_pilots = int(0.9 * params['N'])
    dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

    pareto_results = {}

    for scenario_name, scenario in scenarios.items():
        print(f"\nProcessing {scenario_name} scenario (λ_b={scenario['r_b']})...")

        params_sim = params.copy()
        params_sim['r_b'] = scenario['r_b']
        pareto_points = []

        for D_max in tqdm(D_targets, desc=f"{scenario_name}"):
            max_rate = 0.0

            for alpha in alpha_search:
                for rho in rho_search:
                    try:
                        # Check constraints
                        if params['tau_d']:
                            Smax_eff = min(params['Smax'],
                                           (params['dt'] / params['tau_d']) * params['M_pixels'])
                        else:
                            Smax_eff = params['Smax']

                        A_pilot_use = min(Smax_eff, 4.0 * params['Sbar']) * 0.5

                        S_pilot = rho * params['Sbar'] / alpha
                        S_data = (1 - rho) * params['Sbar'] / (1 - alpha)

                        if S_pilot > Smax_eff or S_data > Smax_eff:
                            continue

                        # Compute FIM
                        I_pilot = fim_pilot(alpha, rho, params['Sbar'], params['N'],
                                            params['dt'], params_sim, dither_seq,
                                            params['tau_d'], A_pilot_use, params['M_pixels'])
                        J = I_pilot + params['J_P']

                        if np.linalg.cond(J) > 1e14:
                            continue

                        J_inv = np.linalg.inv(J)
                        mse_current = np.trace(params['W'] @ J_inv)

                        if mse_current > D_max:
                            continue

                        # Compute rate
                        C_data, _ = capacity_lb(S_data, params['Smax'], scenario['r_b'],
                                                params['dt'], params['tau_d'], params['M_pixels'])
                        rate = (1 - alpha) * C_data

                        if rate > max_rate:
                            max_rate = rate
                    except:
                        continue

            if max_rate > 0:
                pareto_points.append((max_rate, D_max))

        pareto_results[scenario_name] = pareto_points
        print(f"  Found {len(pareto_points)} Pareto points")

    # Plot
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

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
                      label=f'{scenario["name"]}')

    ax.set_xlabel('Mean Squared Error', fontsize=10, weight='bold')
    ax.set_ylabel('Rate [bits/slot]', fontsize=10, weight='bold')
    ax.set_title(f'Rate-MSE Pareto Boundary\n(S̄={params["Sbar"]:.1f}, M={params["M_pixels"]})',
                 fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/rate_mse_boundary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/rate_mse_boundary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: rate_mse_boundary.pdf/png")
    return pareto_results


# ============================================================================
# MAIN DRIVER WITH PRESET SELECTION
# ============================================================================

def main(preset='moderate'):
    """
    Main simulation with preset selection.

    Presets:
    --------
    'low_photon'  : Low budget, strong background effect
    'moderate'    : Balanced, good for all figures (DEFAULT)
    'high_photon' : High budget, showcases array parallelism
    """
    print("\n" + "=" * 80)
    print("OISL-ISAC PERFORMANCE LIMITS SIMULATION (FIXED VERSION)")
    print("All patches applied: ✓ Multi-pixel ✓ α-ρ decoupling ✓ Robust plots ✓ gammaln")
    print("=" * 80)

    # Load preset
    params, scenarios = get_preset_params(preset)

    print(f"\n✓ Using preset: '{preset}'")
    print(f"\nKey Parameters:")
    print(f"  S̄ = {params['Sbar']:.1f} photons/slot")
    print(f"  S_max = {params['Smax']:.1f} photons/slot")
    print(f"  M_pixels = {params['M_pixels']}")
    print(f"  θ_b = {params['theta_b'] * 1e6:.1f} μrad")
    print(f"  τ_d = {params['tau_d'] * 1e9:.1f} ns" if params['tau_d'] else "  τ_d = None")

    # Feasibility check
    if params['tau_d']:
        Smax_eff = min(params['Smax'],
                       (params['dt'] / params['tau_d']) * params['M_pixels'])
        print(f"  S_max_eff = {Smax_eff:.1f} (dead-time limited)")

        if params['Sbar'] > Smax_eff:
            print(f"\n❌ CRITICAL: Sbar > Smax_eff!")
            print(f"   Solutions:")
            print(f"     1. Increase M_pixels to {int(np.ceil(params['Sbar'] * params['tau_d'] / params['dt']))}")
            print(f"     2. Increase dt to {params['tau_d'] * params['Sbar']:.2e} s")
            print(f"     3. Use a different preset")
            return False

    output_dir = f'./results_{preset}'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Generate figures
        print(f"\n{'=' * 60}")
        rb_array, capacity_values = generate_fig_capacity_vs_background(
            params, output_dir, verbose=False)

        print(f"\n{'=' * 60}")
        alpha_range, rho_range, mse_trace = generate_fig_fim_vs_resources(
            params, scenarios, output_dir, verbose=False)

        if mse_trace is not None:
            print(f"\n{'=' * 60}")
            pareto_results = generate_fig_pareto_boundary(
                params, scenarios, output_dir, verbose=False)

        print(f"\n{'=' * 80}")
        print(f"✓ SIMULATION COMPLETED SUCCESSFULLY")
        print(f"{'=' * 80}")
        print(f"Results saved to: {output_dir}/")
        print(f"\nGenerated files:")
        for f in os.listdir(output_dir):
            print(f"  • {f}")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import sys

    # Command line preset selection
    preset = sys.argv[1] if len(sys.argv) > 1 else 'moderate'

    if preset not in ['low_photon', 'moderate', 'high_photon']:
        print(f"Invalid preset: {preset}")
        print("Available presets: low_photon, moderate, high_photon")
        sys.exit(1)

    success = main(preset)
    exit(0 if success else 1)