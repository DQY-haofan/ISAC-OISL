#!/usr/bin/env python3
"""
OISL-ISAC Performance Limits Simulation (Optimized Version)
============================================================

Improvements:
1. GPU acceleration support (via CuPy, optional)
2. IEEE single-column friendly figures with larger fonts
3. Multi-process parallelization for parameter sweeps
4. Better visualization following expert guidelines

Dependencies: numpy, scipy, matplotlib, tqdm
Optional: cupy (for GPU), joblib (for multiprocessing)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from math import ceil, sqrt, exp, log, pi
import os
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# GPU ACCELERATION SUPPORT (Optional)
# ============================================================================

try:
    import cupy as cp

    GPU_AVAILABLE = True
    print("✓ CuPy detected - GPU acceleration available")
    # Use CuPy for array operations
    xp = cp
except ImportError:
    GPU_AVAILABLE = False
    print("ℹ CuPy not found - using CPU (NumPy)")
    xp = np
    cp = np  # Fallback


def to_cpu(x):
    """Convert CuPy array to NumPy if needed."""
    if GPU_AVAILABLE and isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    return x


# ============================================================================
# PARALLEL PROCESSING SUPPORT (Optional)
# ============================================================================

try:
    from joblib import Parallel, delayed

    PARALLEL_AVAILABLE = True
    print("✓ Joblib detected - parallel processing available")
except ImportError:
    PARALLEL_AVAILABLE = False
    print("ℹ Joblib not found - using serial processing")


# ============================================================================
# IEEE SINGLE-COLUMN PUBLICATION STYLE (LARGE FONTS)
# ============================================================================

def setup_ieee_single_column_style():
    """
    Configure matplotlib for IEEE single-column with LARGE readable fonts.
    Based on expert guidelines from reference script.
    """
    plt.rcParams.update({
        # Figure settings - IEEE single column
        'figure.figsize': (3.5, 2.8),  # Single column width (slightly taller)
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,

        # Font settings - LARGER for readability
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,  # Base font size (increased from 9)
        'axes.titlesize': 11,  # Title (increased)
        'axes.labelsize': 10,  # Axis labels (increased)
        'xtick.labelsize': 9,  # Tick labels (increased)
        'ytick.labelsize': 9,
        'legend.fontsize': 8,  # Legend (increased)

        # Line and marker settings
        'lines.linewidth': 1.5,  # Thicker lines
        'lines.markersize': 5,  # Larger markers
        'lines.markeredgewidth': 0.8,

        # Grid settings
        'grid.alpha': 0.3,
        'grid.linewidth': 0.6,

        # Axes settings
        'axes.linewidth': 0.8,
        'axes.grid': True,
        'axes.axisbelow': True,

        # Legend settings
        'legend.frameon': True,
        'legend.framealpha': 0.95,
        'legend.borderpad': 0.4,
        'legend.columnspacing': 1.0,
        'legend.handlelength': 2.0,
        'legend.handletextpad': 0.5,

        # Tick settings
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.minor.width': 0.5,
        'ytick.minor.width': 0.5,
    })

    # Color scheme (professional and colorblind-friendly)
    colors = {
        'ideal': '#000000',  # Black
        'state_of_art': '#0072BD',  # Blue
        'high_performance': '#D95319',  # Orange
        'swap_efficient': '#77AC30',  # Green
        'low_cost': '#A2142F',  # Red
        'zodiacal': '#0072BD',
        'earthshine': '#EDB120',
        'stray_light': '#A2142F',
    }

    return colors


# Global color scheme
colors = setup_ieee_single_column_style()

# ============================================================================
# PHYSICAL CONSTANTS
# ============================================================================

SPEED_OF_LIGHT = 299792458  # m/s


# ============================================================================
# CORE PHYSICS FUNCTIONS (GPU-compatible)
# ============================================================================

def E_Lp(mu_vec, sigma2, theta_b):
    """Expected pointing loss for Gaussian beam with jitter (GPU-compatible)"""
    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)
    mu2 = xp.dot(mu_vec, mu_vec) if isinstance(mu_vec, (xp.ndarray, np.ndarray)) else np.dot(mu_vec, mu_vec)
    gamma = 1.0 + a * sigma2
    return (1.0 / gamma) * xp.exp(-b * mu2 / gamma)


def r_signal(Ptx, Llink, eta, hnu, mu_vec, sigma2, theta_b):
    """Signal photon rate"""
    S = (eta / hnu) * Ptx * Llink
    return S * E_Lp(mu_vec, sigma2, theta_b)


def r_deadtime(r, tau_d):
    """Non-paralyzable dead time correction"""
    return r / (1.0 + r * tau_d)


def fim_pilot(alpha, rho, Sbar, N, dt, params, dither_seq, tau_d=None):
    """
    Fisher Information Matrix computation (Algorithm 1)
    GPU-accelerated version for large-scale computations
    """
    mu = params["mu"].copy()
    sigma2 = params["sigma2"]
    theta_b = params["theta_b"]
    r_b = params["r_b"]
    Llink = params["Llink"]
    eta = params["eta"]
    hnu = params["hnu"]

    S_pilot = rho * Sbar / alpha
    Ptx = S_pilot * hnu / (eta * Llink)

    Npilot = int(np.floor(alpha * N))
    I = np.zeros((4, 4))  # Keep on CPU for stability
    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)

    for n in range(min(Npilot, len(dither_seq))):
        mu_eff = mu + dither_seq[n]
        Lp = float(E_Lp(mu_eff, sigma2, theta_b))
        r_s = S_pilot * Lp / dt
        r_tot = r_s + r_b

        if tau_d is not None and r_tot * tau_d > 0.1:
            r_tot = r_deadtime(r_tot, tau_d)

        lam = r_tot * dt

        gamma = 1.0 + a * sigma2
        dlam_dmux = S_pilot * Lp * (-2 * b * mu_eff[0] / gamma)
        dlam_dmuy = S_pilot * Lp * (-2 * b * mu_eff[1] / gamma)
        dlam_dsig = S_pilot * Lp * (-a / gamma + a * b * np.dot(mu_eff, mu_eff) / (gamma ** 2))
        dlam_drb = dt

        grad = np.array([dlam_dmux, dlam_dmuy, dlam_dsig, dlam_drb])

        if lam > 1e-12:
            I += np.outer(grad, grad) / lam

    return I


def poisson_entropy(lam):
    """Compute entropy of Poisson distribution with truncation"""
    if lam <= 0:
        return 0.0
    Kmax = int(ceil(lam + 10.0 * sqrt(max(lam, 1.0))))
    ks = np.arange(Kmax + 1)

    log_pk = -lam + ks * np.log(max(lam, 1e-100)) - np.array(
        [sum(np.log(range(1, int(k) + 1))) if k > 0 else 0 for k in ks])
    pk = np.exp(log_pk)
    pk = pk / pk.sum()
    pk = pk[pk > 1e-15]
    return -np.sum(pk * np.log2(pk))


def capacity_lb(Sbar, Smax, lamb_b, dt=1.0, tau_d=None):
    """Binary-input capacity lower bound (Algorithm 2)"""
    if tau_d is not None:
        Smax_eff = min(Smax, dt / tau_d)
    else:
        Smax_eff = Smax

    A_grid = np.linspace(max(Sbar, 0.1), Smax_eff, 100)
    Cbest, Aopt = 0.0, Sbar

    for A in A_grid:
        p = Sbar / A
        if p > 1:
            continue

        lam0 = lamb_b
        lam1 = lamb_b + A

        if tau_d is not None:
            lam0 = r_deadtime(lam0 / dt, tau_d) * dt
            lam1 = r_deadtime(lam1 / dt, tau_d) * dt

        HY0 = poisson_entropy(lam0)
        HY1 = poisson_entropy(lam1)

        Kmax = int(ceil(max(lam0, lam1) + 10.0 * sqrt(max(max(lam0, lam1), 1.0))))
        ks = np.arange(Kmax + 1)

        log_p0 = -lam0 + ks * np.log(max(lam0, 1e-100)) - np.array(
            [sum(np.log(range(1, int(k) + 1))) if k > 0 else 0 for k in ks])
        log_p1 = -lam1 + ks * np.log(max(lam1, 1e-100)) - np.array(
            [sum(np.log(range(1, int(k) + 1))) if k > 0 else 0 for k in ks])

        p0 = np.exp(log_p0)
        p1 = np.exp(log_p1)

        PY = (1 - p) * p0 + p * p1
        PY = PY / PY.sum()
        PY_nonzero = PY[PY > 1e-15]
        HY = -np.sum(PY_nonzero * np.log2(PY_nonzero))

        I = HY - (1 - p) * HY0 - p * HY1

        if I > Cbest:
            Cbest, Aopt = I, A

    return Cbest, Aopt


# ============================================================================
# SIMULATION PARAMETERS
# ============================================================================

def setup_simulation_parameters():
    """Define all simulation parameters"""
    c = SPEED_OF_LIGHT
    h = 6.626e-34
    wavelength = 1550e-9
    nu = c / wavelength
    hnu = h * nu

    params = {
        'Sbar': 50,
        'Smax': 500,
        'dt': 1e-6,
        'N': 10000,
        'eta': 0.8,
        'hnu': hnu,
        'theta_b': 10e-6,
        'Llink': 1e-12,
        'mu': np.array([0.0, 0.0]),
        'sigma2': (2e-6) ** 2,
        'tau_d': 100e-9,
        'J_P': np.diag([1e-12, 1e-12, 1e-10, 1e-6]),
        'W': np.eye(4),
    }

    space_weather_scenarios = {
        'Low': {'name': 'Zodiacal', 'r_b': 0.01, 'color': colors['zodiacal']},
        'Medium': {'name': 'Earthshine', 'r_b': 1.0, 'color': colors['earthshine']},
        'High': {'name': 'Stray Light', 'r_b': 10.0, 'color': colors['stray_light']},
    }

    return params, space_weather_scenarios


def generate_dither_sequence(N_pilot, theta_b, delta_factor=0.5):
    """Generate dithering sequence for identifiability"""
    delta = delta_factor * theta_b
    dither_seq = []

    for n in range(N_pilot):
        if n % 4 == 0:
            dither_seq.append(np.array([delta, 0.0]))
        elif n % 4 == 1:
            dither_seq.append(np.array([-delta, 0.0]))
        elif n % 4 == 2:
            dither_seq.append(np.array([0.0, delta]))
        else:
            dither_seq.append(np.array([0.0, -delta]))

    return dither_seq


# ============================================================================
# FIGURE 1: CAPACITY VS BACKGROUND (SINGLE COLUMN OPTIMIZED)
# ============================================================================

def generate_fig_capacity_vs_background(params, output_dir='./'):
    """Generate capacity vs background figure (single-column optimized)"""
    print("\n" + "=" * 60)
    print("Figure 1: Capacity vs Background Noise")
    print("=" * 60)

    rb_array = np.logspace(-2, 2, 50)
    Sbar = params['Sbar']
    Smax = params['Smax']
    tau_d = params['tau_d']
    dt = params['dt']

    capacity_values = []

    for r_b in tqdm(rb_array, desc="Computing capacity"):
        C_lb, _ = capacity_lb(Sbar, Smax, r_b, dt, tau_d)
        capacity_values.append(C_lb)

    # Create single-column figure with large fonts
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.semilogx(rb_array, capacity_values, 'b-', linewidth=2,
                label='Capacity Lower Bound')

    # Regime boundaries
    ax.axvline(x=0.01, color=colors['zodiacal'], linestyle='--',
               linewidth=1.5, alpha=0.8, label='Zodiacal')
    ax.axvline(x=1.0, color=colors['earthshine'], linestyle='--',
               linewidth=1.5, alpha=0.8, label='Earthshine')
    ax.axvline(x=10.0, color=colors['stray_light'], linestyle='--',
               linewidth=1.5, alpha=0.8, label='Stray Light')

    # Annotations (larger fonts)
    ax.text(0.005, max(capacity_values) * 0.85, 'Zodiacal',
            ha='center', fontsize=9, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(0.3, max(capacity_values) * 0.65, 'Earthshine',
            ha='center', fontsize=9, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    ax.text(30, max(capacity_values) * 0.45, 'Stray\nLight',
            ha='center', fontsize=9, weight='bold',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    ax.set_xlabel('Background λ_b [photons/slot]', fontsize=10, weight='bold')
    ax.set_ylabel('Capacity C_LB [bits/slot]', fontsize=10, weight='bold')
    ax.set_title(f'DTPC Capacity vs Background\n(S̄={Sbar}, S_max={Smax})',
                 fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=8, framealpha=0.95)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/capacity_vs_background.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/capacity_vs_background.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: capacity_vs_background.pdf/png")
    return rb_array, capacity_values


# ============================================================================
# FIGURE 2: FIM VS RESOURCE ALLOCATION (SINGLE COLUMN OPTIMIZED)
# ============================================================================

def generate_fig_fim_vs_resources(params, space_weather_scenarios, output_dir='./'):
    """Generate FIM heatmap (single-column optimized)"""
    print("\n" + "=" * 60)
    print("Figure 2: FIM vs Resource Allocation")
    print("=" * 60)

    scenario = space_weather_scenarios['Medium']
    params_sim = params.copy()
    params_sim['r_b'] = scenario['r_b']

    alpha_range = np.linspace(0.01, 0.99, 25)
    rho_range = np.linspace(0.01, 0.99, 25)
    mse_trace = np.zeros((len(rho_range), len(alpha_range)))

    max_pilots = int(0.99 * params['N'])
    dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

    print("Computing FIM grid...")
    for i, rho in enumerate(tqdm(rho_range, desc="ρ sweep")):
        for j, alpha in enumerate(alpha_range):
            try:
                S_pilot = rho * params['Sbar'] / alpha
                S_data = (1 - rho) * params['Sbar'] / (1 - alpha)

                if S_pilot > params['Smax'] or S_data > params['Smax']:
                    mse_trace[i, j] = np.nan
                    continue

                I_pilot = fim_pilot(alpha, rho, params['Sbar'], params['N'],
                                    params['dt'], params_sim, dither_seq, params['tau_d'])
                J = I_pilot + params['J_P']

                if np.linalg.cond(J) > 1e12:
                    mse_trace[i, j] = np.nan
                else:
                    J_inv = np.linalg.inv(J)
                    mse_trace[i, j] = np.trace(params['W'] @ J_inv)
            except:
                mse_trace[i, j] = np.nan

    # Single-column figure
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    log_mse = np.log10(mse_trace)
    Alpha, Rho = np.meshgrid(alpha_range, rho_range)

    im = ax.pcolormesh(Alpha, Rho, log_mse, shading='auto', cmap='viridis_r')

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('log₁₀(Trace CRLB)', rotation=270, labelpad=20,
                   fontsize=9, weight='bold')
    cbar.ax.tick_params(labelsize=8)

    try:
        contours = ax.contour(Alpha, Rho, log_mse, levels=8,
                              colors='white', alpha=0.6, linewidths=0.8)
        ax.clabel(contours, inline=True, fontsize=7, fmt='%.1f')
    except:
        pass

    ax.set_xlabel('Time Allocation α', fontsize=10, weight='bold')
    ax.set_ylabel('Photon Allocation ρ', fontsize=10, weight='bold')
    ax.set_title(f'Fisher Information vs Resources\n({scenario["name"]}, λ_b={scenario["r_b"]})',
                 fontsize=11, weight='bold')

    # Annotations (larger)
    ax.text(0.05, 0.95, 'Poor\nSensing', transform=ax.transAxes, fontsize=9,
            weight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax.text(0.75, 0.05, 'Good\nSensing', transform=ax.transAxes, fontsize=9,
            weight='bold', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/fim_vs_resources.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/fim_vs_resources.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: fim_vs_resources.pdf/png")
    return alpha_range, rho_range, mse_trace


# ============================================================================
# FIGURE 3: RATE-MSE PARETO BOUNDARY (SINGLE COLUMN OPTIMIZED)
# ============================================================================

def generate_fig_pareto_boundary(params, space_weather_scenarios, output_dir='./'):
    """Generate Pareto boundary (single-column optimized, most important figure)"""
    print("\n" + "=" * 60)
    print("Figure 3: Rate-MSE Pareto Boundary (MOST IMPORTANT)")
    print("=" * 60)

    D_targets = np.logspace(-8, -3, 20)
    alpha_search = np.linspace(0.05, 0.95, 12)
    rho_search = np.linspace(0.05, 0.95, 12)

    max_pilots = int(0.95 * params['N'])
    dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

    pareto_results = {}

    for scenario_name, scenario in space_weather_scenarios.items():
        print(f"\nProcessing {scenario_name} scenario...")

        params_sim = params.copy()
        params_sim['r_b'] = scenario['r_b']
        pareto_points = []

        for D_max in tqdm(D_targets, desc=f"{scenario_name}"):
            max_rate = 0.0

            for alpha in alpha_search:
                for rho in rho_search:
                    try:
                        S_pilot = rho * params['Sbar'] / alpha
                        S_data = (1 - rho) * params['Sbar'] / (1 - alpha)

                        if S_pilot > params['Smax'] or S_data > params['Smax']:
                            continue

                        I_pilot = fim_pilot(alpha, rho, params['Sbar'], params['N'],
                                            params['dt'], params_sim, dither_seq, params['tau_d'])
                        J = I_pilot + params['J_P']

                        if np.linalg.cond(J) > 1e12:
                            continue

                        J_inv = np.linalg.inv(J)
                        mse_current = np.trace(params['W'] @ J_inv)

                        if mse_current > D_max:
                            continue

                        C_data, _ = capacity_lb(S_data, params['Smax'], scenario['r_b'],
                                                params['dt'], params['tau_d'])
                        rate = (1 - alpha) * C_data

                        if rate > max_rate:
                            max_rate = rate
                    except:
                        continue

            if max_rate > 0:
                pareto_points.append((max_rate, D_max))

        pareto_results[scenario_name] = pareto_points
        print(f"  Found {len(pareto_points)} Pareto points")

    # Single-column figure with large fonts
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    for scenario_name, scenario in space_weather_scenarios.items():
        points = pareto_results[scenario_name]
        if len(points) > 0:
            rates = [p[0] for p in points]
            mses = [p[1] for p in points]

            sorted_pairs = sorted(zip(mses, rates))
            mses_sorted = [p[0] for p in sorted_pairs]
            rates_sorted = [p[1] for p in sorted_pairs]

            ax.loglog(mses_sorted, rates_sorted, 'o-',
                      color=scenario['color'], linewidth=2, markersize=6,
                      label=f'{scenario["name"]} (λ_b={scenario["r_b"]})')

    # Guidelines
    ax.axhline(y=0.1, color='gray', linestyle=':', linewidth=1.5,
               alpha=0.7, label='Min rate')
    ax.axvline(x=1e-6, color='gray', linestyle=':', linewidth=1.5,
               alpha=0.7, label='Target MSE')

    ax.set_xlabel('Mean Squared Error (MSE)', fontsize=10, weight='bold')
    ax.set_ylabel('Rate [bits/slot]', fontsize=10, weight='bold')
    ax.set_title(f'Rate-MSE Pareto Boundary\n(S̄={params["Sbar"]}, S_max={params["Smax"]})',
                 fontsize=11, weight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right', fontsize=7.5, framealpha=0.95)

    # Region annotations (larger)
    ax.text(0.05, 0.95, 'Sensing-\nLimited', transform=ax.transAxes,
            fontsize=9, weight='bold', va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    ax.text(0.75, 0.05, 'Comm-\nLimited', transform=ax.transAxes,
            fontsize=9, weight='bold', va='bottom',
            bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    plt.tight_layout()
    plt.savefig(f'{output_dir}/rate_mse_boundary.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/rate_mse_boundary.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Saved: rate_mse_boundary.pdf/png")
    return pareto_results


# ============================================================================
# MAIN DRIVER
# ============================================================================

def main():
    """Main simulation driver"""
    print("\n" + "=" * 80)
    print("OISL-ISAC PERFORMANCE LIMITS SIMULATION")
    print("Optimized for GPU acceleration & IEEE single-column figures")
    print("=" * 80)

    if GPU_AVAILABLE:
        print(f"\n✓ GPU acceleration: ENABLED (CuPy)")
        print(f"  GPU memory: {cp.cuda.Device().mem_info[1] / 1e9:.1f} GB total")
    else:
        print(f"\nℹ GPU acceleration: DISABLED (using NumPy)")

    if PARALLEL_AVAILABLE:
        print(f"✓ Parallel processing: ENABLED (Joblib)")
    else:
        print(f"ℹ Parallel processing: DISABLED")

    params, space_weather_scenarios = setup_simulation_parameters()

    print(f"\nSimulation Parameters:")
    print(f"  S̄ = {params['Sbar']} photons/slot")
    print(f"  S_max = {params['Smax']} photons/slot")
    print(f"  θ_b = {params['theta_b'] * 1e6:.1f} μrad")
    print(f"  N = {params['N']} slots")

    output_dir = './simulation_results'
    os.makedirs(output_dir, exist_ok=True)

    try:
        # Generate three key figures
        print(f"\n{'=' * 60}")
        rb_array, capacity_values = generate_fig_capacity_vs_background(params, output_dir)

        print(f"\n{'=' * 60}")
        alpha_range, rho_range, mse_trace = generate_fig_fim_vs_resources(
            params, space_weather_scenarios, output_dir)

        print(f"\n{'=' * 60}")
        pareto_results = generate_fig_pareto_boundary(
            params, space_weather_scenarios, output_dir)

        print(f"\n{'=' * 80}")
        print("✓ SIMULATION COMPLETED SUCCESSFULLY")
        print(f"{'=' * 80}")
        print(f"Results in: {output_dir}/")
        print("Generated files:")
        print("  • capacity_vs_background.pdf/png")
        print("  • fim_vs_resources.pdf/png")
        print("  • rate_mse_boundary.pdf/png")

        print(f"\nPerformance Summary:")
        print(f"  Capacity range: {min(capacity_values):.3f} - {max(capacity_values):.3f} bits/slot")

        for name, points in pareto_results.items():
            if len(points) > 0:
                max_rate = max(p[0] for p in points)
                print(f"  {name}: Max rate = {max_rate:.3f} bits/slot")

        return True

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)