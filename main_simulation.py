#!/usr/bin/env python3
"""
OISL-ISAC Performance Limits Simulation
========================================

This script implements the fundamental performance limits for Optical Inter-Satellite Link
Integrated Sensing and Communication (OISL-ISAC) systems under space weather influences.

Based on the theoretical framework from:
- Section II: System Model and Problem Formulation
- Section III: Fundamental Performance Limits

Generates three key performance figures:
1. Capacity vs. Background Noise
2. Fisher Information vs. Resource Allocation (Heatmap)
3. Rate-MSE Pareto Boundary (Most Important)

Author: Generated from academic paper implementation
Dependencies: numpy, scipy, matplotlib
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
# CORE PHYSICS AND ALGORITHM FUNCTIONS (from isac_core.py)
# ============================================================================

def E_Lp(mu_vec, sigma2, theta_b):
    """Expected pointing loss for Gaussian beam with jitter"""
    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)
    mu2 = np.dot(mu_vec, mu_vec)
    gamma = 1.0 + a * sigma2
    return (1.0 / gamma) * np.exp(-b * mu2 / gamma)


def r_signal(Ptx, Llink, eta, hnu, mu_vec, sigma2, theta_b):
    """Signal photon rate"""
    S = (eta / hnu) * Ptx * Llink
    return S * E_Lp(mu_vec, sigma2, theta_b)


def r_deadtime(r, tau_d):
    """Non-paralyzable dead time correction"""
    return r / (1.0 + r * tau_d)


def fim_pilot(alpha, rho, Sbar, N, dt, params, dither_seq, tau_d=None):
    """
    Fisher Information Matrix computation for pilot signals
    Algorithm 1 from Section III
    """
    mu = params["mu"].copy()
    sigma2 = params["sigma2"]
    theta_b = params["theta_b"]
    r_b = params["r_b"]
    Llink = params["Llink"]
    eta = params["eta"]
    hnu = params["hnu"]

    # Pilot power calculation
    S_pilot = rho * Sbar / alpha
    Ptx = S_pilot * hnu / (eta * Llink)

    Npilot = int(np.floor(alpha * N))
    I = np.zeros((4, 4))
    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)

    for n in range(min(Npilot, len(dither_seq))):
        mu_eff = mu + dither_seq[n]  # Apply dithering for identifiability
        Lp = E_Lp(mu_eff, sigma2, theta_b)
        r_s = S_pilot * Lp / dt  # Signal rate
        r_tot = r_s + r_b

        if tau_d is not None and r_tot * tau_d > 0.1:
            r_tot = r_deadtime(r_tot, tau_d)

        lam = r_tot * dt

        # Partial derivatives
        gamma = 1.0 + a * sigma2
        dlam_dmux = S_pilot * Lp * (-2 * b * mu_eff[0] / gamma)
        dlam_dmuy = S_pilot * Lp * (-2 * b * mu_eff[1] / gamma)
        dlam_dsig = S_pilot * Lp * (-a / gamma + a * b * np.dot(mu_eff, mu_eff) / (gamma ** 2))
        dlam_drb = dt

        grad = np.array([dlam_dmux, dlam_dmuy, dlam_dsig, dlam_drb])

        if lam > 1e-12:  # Numerical stability
            I += np.outer(grad, grad) / lam

    return I


def poisson_entropy(lam):
    """Compute entropy of Poisson distribution with truncation"""
    if lam <= 0:
        return 0.0
    Kmax = int(ceil(lam + 10.0 * sqrt(max(lam, 1.0))))
    ks = np.arange(Kmax + 1)

    # Use log-space computation for numerical stability
    log_pk = -lam + ks * np.log(max(lam, 1e-100)) - np.array(
        [sum(np.log(range(1, int(k) + 1))) if k > 0 else 0 for k in ks])
    pk = np.exp(log_pk)
    pk = pk / pk.sum()

    # Remove zero probabilities
    pk = pk[pk > 1e-15]
    return -np.sum(pk * np.log2(pk))


def capacity_lb(Sbar, Smax, lamb_b, dt=1.0, tau_d=None):
    """
    Binary-input capacity lower bound
    Algorithm 2 from Section III
    """
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

        # Apply dead time if specified
        if tau_d is not None:
            lam0 = r_deadtime(lam0 / dt, tau_d) * dt
            lam1 = r_deadtime(lam1 / dt, tau_d) * dt

        # Compute entropies
        HY0 = poisson_entropy(lam0)
        HY1 = poisson_entropy(lam1)

        # Mixture distribution
        Kmax = int(ceil(max(lam0, lam1) + 10.0 * sqrt(max(max(lam0, lam1), 1.0))))
        ks = np.arange(Kmax + 1)

        # Probability mass functions
        log_p0 = -lam0 + ks * np.log(max(lam0, 1e-100)) - np.array(
            [sum(np.log(range(1, int(k) + 1))) if k > 0 else 0 for k in ks])
        log_p1 = -lam1 + ks * np.log(max(lam1, 1e-100)) - np.array(
            [sum(np.log(range(1, int(k) + 1))) if k > 0 else 0 for k in ks])

        p0 = np.exp(log_p0)
        p1 = np.exp(log_p1)

        PY = (1 - p) * p0 + p * p1
        PY = PY / PY.sum()

        # Remove zero probabilities for entropy calculation
        PY_nonzero = PY[PY > 1e-15]
        HY = -np.sum(PY_nonzero * np.log2(PY_nonzero))

        # Mutual information
        I = HY - (1 - p) * HY0 - p * HY1

        if I > Cbest:
            Cbest, Aopt = I, A

    return Cbest, Aopt


# ============================================================================
# SIMULATION PARAMETERS SETUP
# ============================================================================

def setup_simulation_parameters():
    """
    Define all simulation parameters based on Section II Table and Section III examples
    """
    # Physical constants
    c = 3e8  # Speed of light [m/s]
    h = 6.626e-34  # Planck constant [J⋅s]
    wavelength = 1550e-9  # Wavelength [m]
    nu = c / wavelength  # Frequency [Hz]
    hnu = h * nu  # Single photon energy [J]

    # System parameters from paper
    params = {
        # Photon budget and constraints
        'Sbar': 50,  # Average signal photons per slot
        'Smax': 500,  # Peak photons per slot constraint
        'dt': 1e-6,  # Time slot duration [s]
        'N': 10000,  # Total number of slots

        # Optical system
        'eta': 0.8,  # Detector quantum efficiency
        'hnu': hnu,  # Single photon energy [J]
        'theta_b': 10e-6,  # Beam divergence half-angle [rad]
        'Llink': 1e-12,  # Static link budget factor

        # Pointing parameters
        'mu': np.array([0.0, 0.0]),  # Mean pointing error [rad]
        'sigma2': (2e-6) ** 2,  # Pointing jitter variance [rad²]

        # Dead time (for SPADs)
        'tau_d': 100e-9,  # Dead time [s]

        # Prior information for BCRLB
        'J_P': np.diag([1e-12, 1e-12, 1e-10, 1e-6]),  # Prior Fisher information
        'W': np.eye(4),  # Weighting matrix for MSE
    }

    # Space weather scenarios from Section II
    space_weather_scenarios = {
        'Low': {
            'name': 'Zodiacal Light Dominated',
            'r_b': 0.01,  # photons per slot
            'DDD': 0.0,  # MeV/g (beginning of life)
            'color': 'blue',
            'description': 'Deep space, minimal background'
        },
        'Medium': {
            'name': 'Earthshine Dominated',
            'r_b': 1.0,  # photons per slot
            'DDD': 1e10,  # MeV/g (moderate radiation)
            'color': 'orange',
            'description': 'Earth-pointing, moderate background'
        },
        'High': {
            'name': 'Stray Light / High Radiation',
            'r_b': 10.0,  # photons per slot
            'DDD': 1e11,  # MeV/g (high radiation environment)
            'color': 'red',
            'description': 'Near-sun or high radiation event'
        }
    }

    return params, space_weather_scenarios


def generate_dither_sequence(N_pilot, theta_b, delta_factor=0.5):
    """
    Generate dithering sequence for parameter identifiability
    Alternates between [±δ, 0] and [0, ±δ] to ensure full rank gradient matrix
    """
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
# FIGURE 1: CAPACITY VS BACKGROUND NOISE
# ============================================================================

def generate_fig_capacity_vs_background(params, output_dir='./'):
    """
    Generate Figure: Capacity lower bound vs. background photon count
    Shows logarithmic degradation with increasing background
    """
    print("Generating Figure 1: Capacity vs Background Noise...")

    # Background noise range (log scale)
    rb_array = np.logspace(-2, 2, 50)  # 0.01 to 100 photons/slot

    # Fixed parameters
    Sbar = params['Sbar']
    Smax = params['Smax']
    tau_d = params['tau_d']
    dt = params['dt']

    # Compute capacity for each background level
    capacity_values = []
    optimal_A_values = []

    for r_b in tqdm(rb_array, desc="Computing capacity"):
        C_lb, A_opt = capacity_lb(Sbar, Smax, r_b, dt, tau_d)
        capacity_values.append(C_lb)
        optimal_A_values.append(A_opt)

    # Create figure
    plt.figure(figsize=(10, 6))
    plt.semilogx(rb_array, capacity_values, 'b-', linewidth=2, label='Capacity Lower Bound')

    # Mark the three regime boundaries
    plt.axvline(x=0.01, color='blue', linestyle='--', alpha=0.7, label='Zodiacal')
    plt.axvline(x=1.0, color='orange', linestyle='--', alpha=0.7, label='Earthshine')
    plt.axvline(x=10.0, color='red', linestyle='--', alpha=0.7, label='Stray Light')

    # Add regime annotations
    plt.text(0.005, max(capacity_values) * 0.9, 'Zodiacal\nRegime', ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    plt.text(0.3, max(capacity_values) * 0.7, 'Earthshine\nRegime', ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    plt.text(30, max(capacity_values) * 0.5, 'Stray Light\nRegime', ha='center', va='top',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))

    plt.xlabel('Background Photon Count λ_b [photons/slot]')
    plt.ylabel('Capacity Lower Bound C_LB [bits/slot]')
    plt.title(f'DTPC Capacity vs Background Noise\n(S̄={Sbar}, S_max={Smax} photons/slot)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'capacity_vs_background.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()

    return rb_array, capacity_values


# ============================================================================
# FIGURE 2: FISHER INFORMATION VS RESOURCE ALLOCATION
# ============================================================================

def generate_fig_fim_vs_resources(params, space_weather_scenarios, output_dir='./'):
    """
    Generate Figure: Fisher information heatmap vs resource allocation (α, ρ)
    Shows benefit of concentrating photons into fewer pilot slots
    """
    print("Generating Figure 2: FIM vs Resource Allocation...")

    # Use Medium scenario as baseline
    scenario = space_weather_scenarios['Medium']
    params_sim = params.copy()
    params_sim['r_b'] = scenario['r_b']

    # Resource allocation grids
    alpha_range = np.linspace(0.01, 0.99, 30)
    rho_range = np.linspace(0.01, 0.99, 30)

    # Results storage
    mse_trace = np.zeros((len(rho_range), len(alpha_range)))

    # Generate dither sequence
    max_pilots = int(0.99 * params['N'])
    dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

    print("Computing FIM for resource allocation grid...")
    for i, rho in enumerate(tqdm(rho_range, desc="ρ values")):
        for j, alpha in enumerate(alpha_range):
            try:
                # Check feasibility constraints
                S_pilot = rho * params['Sbar'] / alpha
                S_data = (1 - rho) * params['Sbar'] / (1 - alpha)

                if S_pilot > params['Smax'] or S_data > params['Smax']:
                    mse_trace[i, j] = np.nan
                    continue

                # Compute FIM
                I_pilot = fim_pilot(alpha, rho, params['Sbar'], params['N'],
                                    params['dt'], params_sim, dither_seq, params['tau_d'])

                # Add prior and compute trace of inverse (total MSE)
                J = I_pilot + params['J_P']

                # Check if matrix is invertible
                if np.linalg.cond(J) > 1e12:
                    mse_trace[i, j] = np.nan
                else:
                    J_inv = np.linalg.inv(J)
                    mse_trace[i, j] = np.trace(params['W'] @ J_inv)

            except Exception as e:
                mse_trace[i, j] = np.nan

    # Create heatmap
    plt.figure(figsize=(10, 8))

    # Use log scale for better visualization
    log_mse = np.log10(mse_trace)

    # Create meshgrid for plotting
    Alpha, Rho = np.meshgrid(alpha_range, rho_range)

    # Plot heatmap
    im = plt.pcolormesh(Alpha, Rho, log_mse, shading='auto', cmap='viridis_r')

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label('log₁₀(Trace of CRLB Matrix)', rotation=270, labelpad=20)

    # Add contours for better readability
    try:
        contours = plt.contour(Alpha, Rho, log_mse, levels=10, colors='white', alpha=0.5, linewidths=0.5)
        plt.clabel(contours, inline=True, fontsize=8)
    except:
        pass

    plt.xlabel('Time Allocation α')
    plt.ylabel('Photon Allocation ρ')
    plt.title(f'Fisher Information vs Resource Allocation\n'
              f'({scenario["name"]}, λ_b = {scenario["r_b"]} photons/slot)')

    # Add annotations
    plt.text(0.05, 0.95, 'High MSE\n(Poor sensing)', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    plt.text(0.7, 0.05, 'Low MSE\n(Good sensing)', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'fim_vs_resources.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()

    return alpha_range, rho_range, mse_trace


# ============================================================================
# FIGURE 3: RATE-MSE PARETO BOUNDARY (MOST IMPORTANT)
# ============================================================================

def generate_fig_pareto_boundary(params, space_weather_scenarios, output_dir='./'):
    """
    Generate Figure: Rate-MSE Pareto boundary for different space weather scenarios
    Algorithm 3 from Section III - most important figure
    """
    print("Generating Figure 3: Rate-MSE Pareto Boundary...")

    # MSE targets for Pareto tracing
    D_targets = np.logspace(-8, -3, 25)  # Range of MSE targets

    # Resource allocation search grids (coarser for speed)
    alpha_search = np.linspace(0.05, 0.95, 15)
    rho_search = np.linspace(0.05, 0.95, 15)

    # Generate dither sequence
    max_pilots = int(0.95 * params['N'])
    dither_seq = generate_dither_sequence(max_pilots, params['theta_b'])

    pareto_results = {}

    # Process each space weather scenario
    for scenario_name, scenario in space_weather_scenarios.items():
        print(f"\nProcessing {scenario_name} scenario...")

        params_sim = params.copy()
        params_sim['r_b'] = scenario['r_b']

        pareto_points = []

        # Outer loop: sweep MSE targets (Algorithm 3)
        for D_max in tqdm(D_targets, desc=f"{scenario_name} MSE targets"):
            max_rate = 0.0
            best_allocation = None

            # Inner loops: search over resource allocation
            for alpha in alpha_search:
                for rho in rho_search:
                    try:
                        # Check peak constraints
                        S_pilot = rho * params['Sbar'] / alpha
                        S_data = (1 - rho) * params['Sbar'] / (1 - alpha)

                        if S_pilot > params['Smax'] or S_data > params['Smax']:
                            continue

                        # Compute pilot FIM
                        I_pilot = fim_pilot(alpha, rho, params['Sbar'], params['N'],
                                            params['dt'], params_sim, dither_seq, params['tau_d'])

                        J = I_pilot + params['J_P']

                        # Check sensing constraint
                        if np.linalg.cond(J) > 1e12:
                            continue

                        J_inv = np.linalg.inv(J)
                        mse_current = np.trace(params['W'] @ J_inv)

                        if mse_current > D_max:
                            continue  # Does not meet sensing requirement

                        # Compute achievable communication rate
                        C_data, _ = capacity_lb(S_data, params['Smax'], scenario['r_b'],
                                                params['dt'], params['tau_d'])

                        rate = (1 - alpha) * C_data  # Account for pilot time

                        if rate > max_rate:
                            max_rate = rate
                            best_allocation = (alpha, rho)

                    except Exception as e:
                        continue

            # Store Pareto point if valid solution found
            if max_rate > 0:
                pareto_points.append((max_rate, D_max, best_allocation))

        pareto_results[scenario_name] = pareto_points
        print(f"Found {len(pareto_points)} Pareto points for {scenario_name}")

    # Create Pareto boundary plot
    plt.figure(figsize=(12, 8))

    for scenario_name, scenario in space_weather_scenarios.items():
        points = pareto_results[scenario_name]
        if len(points) > 0:
            rates = [p[0] for p in points]
            mses = [p[1] for p in points]

            # Sort by MSE for smooth curve
            sorted_pairs = sorted(zip(mses, rates))
            mses_sorted = [p[0] for p in sorted_pairs]
            rates_sorted = [p[1] for p in sorted_pairs]

            plt.loglog(mses_sorted, rates_sorted, 'o-',
                       color=scenario['color'], linewidth=2, markersize=6,
                       label=f'{scenario_name}: {scenario["description"]}')

    # Add theoretical limits annotations
    plt.axhline(y=0.1, color='gray', linestyle=':', alpha=0.7, label='Minimum useful rate')
    plt.axvline(x=1e-6, color='gray', linestyle=':', alpha=0.7, label='Target sensing precision')

    plt.xlabel('Mean Squared Error (MSE)')
    plt.ylabel('Communication Rate [bits/slot]')
    plt.title('Rate-MSE Pareto Boundary under Space Weather\n' +
              f'(S̄={params["Sbar"]}, S_max={params["Smax"]}, N={params["N"]})')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right')

    # Add design insight annotations
    plt.text(0.05, 0.95, 'Sensing-Limited\nRegion', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7),
             verticalalignment='top')
    plt.text(0.7, 0.05, 'Communication-Limited\nRegion', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7),
             verticalalignment='bottom')

    plt.tight_layout()

    # Save figure
    output_path = os.path.join(output_dir, 'rate_mse_boundary.pdf')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.show()

    return pareto_results


# ============================================================================
# MAIN SIMULATION DRIVER
# ============================================================================

def main():
    """
    Main simulation driver - generates all three key performance figures
    """
    print("=" * 80)
    print("OISL-ISAC PERFORMANCE LIMITS SIMULATION")
    print("=" * 80)

    # Setup simulation parameters
    params, space_weather_scenarios = setup_simulation_parameters()

    print(f"\nSimulation Parameters:")
    print(f"- Average signal budget: {params['Sbar']} photons/slot")
    print(f"- Peak constraint: {params['Smax']} photons/slot")
    print(f"- Beam divergence: {params['theta_b'] * 1e6:.1f} μrad")
    print(f"- Total slots: {params['N']}")
    print(f"- Time slot duration: {params['dt'] * 1e6:.1f} μs")

    print(f"\nSpace Weather Scenarios:")
    for name, scenario in space_weather_scenarios.items():
        print(f"- {name}: {scenario['description']} (λ_b = {scenario['r_b']} photons/slot)")

    # Create output directory
    output_dir = './simulation_results'
    os.makedirs(output_dir, exist_ok=True)

    # Generate figures
    try:
        # Figure 1: Capacity vs Background
        print(f"\n{'-' * 60}")
        rb_array, capacity_values = generate_fig_capacity_vs_background(params, output_dir)

        # Figure 2: FIM vs Resource Allocation
        print(f"\n{'-' * 60}")
        alpha_range, rho_range, mse_trace = generate_fig_fim_vs_resources(
            params, space_weather_scenarios, output_dir)

        # Figure 3: Rate-MSE Pareto Boundary (Most Important)
        print(f"\n{'-' * 60}")
        pareto_results = generate_fig_pareto_boundary(
            params, space_weather_scenarios, output_dir)

        print(f"\n{'=' * 80}")
        print("SIMULATION COMPLETED SUCCESSFULLY")
        print(f"{'=' * 80}")
        print(f"Results saved to: {output_dir}/")
        print("Generated files:")
        print("- capacity_vs_background.pdf")
        print("- fim_vs_resources.pdf")
        print("- rate_mse_boundary.pdf")

        # Summary statistics
        print(f"\nPerformance Summary:")
        min_cap = min(capacity_values)
        max_cap = max(capacity_values)
        print(f"- Capacity range: {min_cap:.3f} - {max_cap:.3f} bits/slot")
        print(f"- Background dynamic range: {min(rb_array):.3f} - {max(rb_array):.1f} photons/slot")

        for scenario_name, points in pareto_results.items():
            if len(points) > 0:
                max_rate = max(p[0] for p in points)
                min_mse = min(p[1] for p in points)
                print(f"- {scenario_name}: Max rate = {max_rate:.3f} bits/slot, Min MSE = {min_mse:.2e}")

    except Exception as e:
        print(f"\nERROR during simulation: {e}")
        import traceback
        traceback.print_exc()
        return False

    return True


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)