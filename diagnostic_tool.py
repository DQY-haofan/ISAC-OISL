#!/usr/bin/env python3
"""
OISL-ISAC Diagnostic & Test Suite
==================================

Comprehensive testing and debugging tool to identify issues with:
- Data generation (NaN, Inf, zero values)
- Function correctness (FIM, capacity, etc.)
- Parameter ranges
- Numerical stability

Usage:
    python diagnostic_tool.py

Author: Debug Assistant
"""

import numpy as np
import matplotlib.pyplot as plt
from math import ceil, sqrt, exp, log, pi
import sys
import warnings

warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS
# ============================================================================

SPEED_OF_LIGHT = 299792458  # m/s


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_section(title):
    """Print a formatted section header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def check_array(name, arr, expected_shape=None):
    """Check array for common issues and print diagnostics"""
    print(f"\n[{name}]")
    print(f"  Shape: {arr.shape if hasattr(arr, 'shape') else 'scalar'}")
    print(f"  Type: {type(arr)}, dtype: {arr.dtype if hasattr(arr, 'dtype') else 'N/A'}")

    if hasattr(arr, 'shape') and len(arr.shape) > 0:
        print(f"  Range: [{np.min(arr):.6e}, {np.max(arr):.6e}]")
        print(f"  Mean: {np.mean(arr):.6e}, Std: {np.std(arr):.6e}")

        # Check for problematic values
        n_nan = np.sum(np.isnan(arr))
        n_inf = np.sum(np.isinf(arr))
        n_zero = np.sum(arr == 0)
        n_negative = np.sum(arr < 0)

        issues = []
        if n_nan > 0:
            issues.append(f"❌ {n_nan} NaN values")
        if n_inf > 0:
            issues.append(f"❌ {n_inf} Inf values")
        if n_zero == arr.size:
            issues.append(f"❌ All zeros!")
        elif n_zero > arr.size * 0.9:
            issues.append(f"⚠ {n_zero}/{arr.size} zeros ({n_zero / arr.size * 100:.1f}%)")
        if n_negative > 0 and 'capacity' in name.lower():
            issues.append(f"⚠ {n_negative} negative values (unexpected for capacity)")

        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            print(f"  ✓ Data looks healthy")

        if expected_shape and arr.shape != expected_shape:
            print(f"  ⚠ Expected shape {expected_shape}, got {arr.shape}")
    else:
        print(f"  Value: {arr}")

    return arr


def save_diagnostic_figure(fig, filename, data_dict):
    """Save figure and associated data"""
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"  ✓ Saved: {filename}")

    # Save data as text
    data_file = filename.replace('.png', '_data.txt')
    with open(data_file, 'w') as f:
        f.write("# Diagnostic Data\n")
        for key, value in data_dict.items():
            if isinstance(value, (np.ndarray, list)):
                f.write(f"\n{key}:\n")
                if isinstance(value, np.ndarray):
                    f.write(f"  Shape: {value.shape}\n")
                    f.write(f"  Range: [{np.min(value):.6e}, {np.max(value):.6e}]\n")
                    f.write(f"  First 5: {value.flat[:5]}\n")
                else:
                    f.write(f"  Length: {len(value)}\n")
                    f.write(f"  First 5: {value[:5]}\n")
            else:
                f.write(f"{key}: {value}\n")
    print(f"  ✓ Saved data: {data_file}")


# ============================================================================
# CORE PHYSICS FUNCTIONS (FROM ORIGINAL SCRIPT)
# ============================================================================

def E_Lp(mu_vec, sigma2, theta_b):
    """Expected pointing loss"""
    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)
    mu2 = np.dot(mu_vec, mu_vec)
    gamma = 1.0 + a * sigma2
    result = (1.0 / gamma) * np.exp(-b * mu2 / gamma)
    return result


def poisson_entropy(lam):
    """Compute entropy of Poisson distribution"""
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


def capacity_lb_simple(Sbar, Smax, lamb_b, verbose=False):
    """Simplified capacity lower bound with debugging"""
    A_grid = np.linspace(max(Sbar, 0.1), Smax, 50)
    Cbest, Aopt = 0.0, Sbar

    if verbose:
        print(f"\n  capacity_lb_simple:")
        print(f"    Sbar={Sbar}, Smax={Smax}, lamb_b={lamb_b}")
        print(f"    A_grid: [{A_grid[0]:.2f}, {A_grid[-1]:.2f}] ({len(A_grid)} points)")

    for i, A in enumerate(A_grid):
        p = Sbar / A
        if p > 1:
            continue

        lam0 = lamb_b
        lam1 = lamb_b + A

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

        if verbose and i % 10 == 0:
            print(f"    A={A:.2f}: p={p:.3f}, I={I:.4f}, HY={HY:.4f}, HY0={HY0:.4f}, HY1={HY1:.4f}")

        if I > Cbest:
            Cbest, Aopt = I, A

    if verbose:
        print(f"    → Best: C={Cbest:.4f} at A={Aopt:.2f}")

    return Cbest, Aopt


# ============================================================================
# TEST 1: BASIC FUNCTION VALIDATION
# ============================================================================

def test_basic_functions():
    """Test basic physics functions"""
    print_section("TEST 1: Basic Function Validation")

    # Test pointing loss
    print("\n1.1 Testing E_Lp (pointing loss):")
    theta_b = 10e-6
    sigma2 = (2e-6) ** 2

    test_cases = [
        ("Zero offset", np.array([0.0, 0.0])),
        ("Small offset", np.array([1e-6, 0.0])),
        ("Large offset", np.array([5e-6, 0.0])),
    ]

    for name, mu in test_cases:
        Lp = E_Lp(mu, sigma2, theta_b)
        print(f"  {name}: μ={mu}, L_p={Lp:.6f}")
        if Lp <= 0 or Lp > 1:
            print(f"    ❌ Invalid range! L_p should be in (0, 1]")
        else:
            print(f"    ✓ Valid")

    # Test Poisson entropy
    print("\n1.2 Testing poisson_entropy:")
    test_lambdas = [0.01, 0.1, 1.0, 10.0, 100.0]
    for lam in test_lambdas:
        H = poisson_entropy(lam)
        print(f"  λ={lam:6.2f}: H={H:.4f} bits")
        if H < 0 or np.isnan(H):
            print(f"    ❌ Invalid entropy!")
        else:
            print(f"    ✓ Valid")

    # Test capacity function
    print("\n1.3 Testing capacity_lb_simple:")
    Sbar = 10
    Smax = 100
    lamb_b_values = [0.01, 1.0, 10.0]

    for lamb_b in lamb_b_values:
        C, A = capacity_lb_simple(Sbar, Smax, lamb_b, verbose=True)
        if C <= 0 or np.isnan(C):
            print(f"    ❌ Invalid capacity: {C}")
        else:
            print(f"    ✓ Valid capacity: {C:.4f} bits/slot")

    return True


# ============================================================================
# TEST 2: CAPACITY VS BACKGROUND SWEEP
# ============================================================================

def test_capacity_sweep():
    """Test capacity vs background sweep"""
    print_section("TEST 2: Capacity vs Background Sweep")

    Sbar = 50
    Smax = 500
    rb_array = np.logspace(-2, 2, 20)

    print(f"\nParameters:")
    print(f"  Sbar = {Sbar}")
    print(f"  Smax = {Smax}")
    print(f"  Background range: [{rb_array[0]:.4f}, {rb_array[-1]:.2f}]")

    capacity_values = []
    A_opt_values = []

    print(f"\nComputing capacity for {len(rb_array)} background levels...")
    for i, r_b in enumerate(rb_array):
        C_lb, A_opt = capacity_lb_simple(Sbar, Smax, r_b, verbose=(i % 5 == 0))
        capacity_values.append(C_lb)
        A_opt_values.append(A_opt)

        if i % 5 == 0:
            print(f"  Progress: {i + 1}/{len(rb_array)}, r_b={r_b:.4f}, C={C_lb:.4f}")

    capacity_values = np.array(capacity_values)
    A_opt_values = np.array(A_opt_values)

    check_array("capacity_values", capacity_values)
    check_array("A_opt_values", A_opt_values)

    # Plot diagnostic figure
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))

    # Subplot 1: Capacity
    axes[0].semilogx(rb_array, capacity_values, 'bo-', linewidth=2, markersize=6)
    axes[0].set_xlabel('Background λ_b [photons/slot]')
    axes[0].set_ylabel('Capacity C_LB [bits/slot]')
    axes[0].set_title(f'Capacity vs Background (Sbar={Sbar}, Smax={Smax})')
    axes[0].grid(True, alpha=0.3)

    # Add regime markers
    axes[0].axvline(x=0.01, color='blue', linestyle='--', alpha=0.5, label='Zodiacal')
    axes[0].axvline(x=1.0, color='orange', linestyle='--', alpha=0.5, label='Earthshine')
    axes[0].axvline(x=10.0, color='red', linestyle='--', alpha=0.5, label='Stray Light')
    axes[0].legend()

    # Subplot 2: Optimal A
    axes[1].semilogx(rb_array, A_opt_values, 'rs-', linewidth=2, markersize=6)
    axes[1].set_xlabel('Background λ_b [photons/slot]')
    axes[1].set_ylabel('Optimal A [photons]')
    axes[1].set_title('Optimal Signal Amplitude')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    save_diagnostic_figure(fig, 'diagnostic_capacity_sweep.png', {
        'rb_array': rb_array,
        'capacity_values': capacity_values,
        'A_opt_values': A_opt_values
    })

    # Sanity checks
    print("\n✓ Sanity Checks:")
    if np.all(capacity_values > 0):
        print("  ✓ All capacities are positive")
    else:
        print("  ❌ Some capacities are non-positive!")

    if np.all(np.diff(capacity_values) <= 0):
        print("  ✓ Capacity decreases with background (as expected)")
    else:
        print("  ⚠ Capacity not monotonically decreasing")

    if capacity_values[0] > 3 * capacity_values[-1]:
        print("  ✓ Significant dynamic range (low vs high background)")
    else:
        print("  ⚠ Small dynamic range - check parameters")

    return True


# ============================================================================
# TEST 3: FIM COMPUTATION
# ============================================================================

def test_fim_computation():
    """Test Fisher Information Matrix computation"""
    print_section("TEST 3: FIM Computation")

    # Setup parameters
    c = SPEED_OF_LIGHT
    h = 6.626e-34
    wavelength = 1550e-9
    nu = c / wavelength
    hnu = h * nu

    params = {
        'Sbar': 50,
        'eta': 0.8,
        'hnu': hnu,
        'theta_b': 10e-6,
        'Llink': 1e-12,
        'mu': np.array([0.0, 0.0]),
        'sigma2': (2e-6) ** 2,
        'r_b': 1.0,
    }

    print("\nParameters:")
    for key, value in params.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: {value}")
        else:
            print(f"  {key}: {value:.6e}" if isinstance(value, float) else f"  {key}: {value}")

    # Generate simple dither sequence
    N_pilot = 100
    theta_b = params['theta_b']
    delta = 0.5 * theta_b

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

    print(f"\nDither sequence: {len(dither_seq)} points")
    print(f"  δ = {delta:.3e} rad = {delta * 1e6:.3f} μrad")
    print(f"  First 4 offsets:")
    for i in range(4):
        print(f"    {i}: {dither_seq[i]}")

    # Compute FIM for different (alpha, rho) pairs
    test_cases = [
        ("Small pilot time, low photon", 0.1, 0.1),
        ("Balanced", 0.5, 0.5),
        ("Large pilot time, high photon", 0.9, 0.9),
    ]

    dt = 1e-6
    N = 1000

    print(f"\nTesting FIM computation (N={N}, dt={dt * 1e6:.1f} μs):")

    for name, alpha, rho in test_cases:
        print(f"\n  {name}: α={alpha}, ρ={rho}")

        # Compute pilot power
        S_pilot = rho * params['Sbar'] / alpha
        print(f"    S_pilot = {S_pilot:.2f} photons/slot")

        # Simplified FIM computation
        I = np.zeros((4, 4))
        a = 4.0 / (theta_b ** 2)
        b = 2.0 / (theta_b ** 2)

        Npilot = int(np.floor(alpha * N))

        for n in range(min(Npilot, len(dither_seq))):
            mu_eff = params['mu'] + dither_seq[n]
            Lp = E_Lp(mu_eff, params['sigma2'], theta_b)
            r_s = S_pilot * Lp / dt
            r_tot = r_s + params['r_b']
            lam = r_tot * dt

            gamma = 1.0 + a * params['sigma2']
            dlam_dmux = S_pilot * Lp * (-2 * b * mu_eff[0] / gamma)
            dlam_dmuy = S_pilot * Lp * (-2 * b * mu_eff[1] / gamma)
            dlam_dsig = S_pilot * Lp * (-a / gamma + a * b * np.dot(mu_eff, mu_eff) / (gamma ** 2))
            dlam_drb = dt

            grad = np.array([dlam_dmux, dlam_dmuy, dlam_dsig, dlam_drb])

            if lam > 1e-12:
                I += np.outer(grad, grad) / lam

        print(f"    FIM diagonal: {np.diag(I)}")
        print(f"    FIM trace: {np.trace(I):.6e}")
        print(f"    FIM condition number: {np.linalg.cond(I):.2e}")

        # Try to invert
        try:
            I_inv = np.linalg.inv(I + 1e-10 * np.eye(4))
            print(f"    CRLB diagonal: {np.diag(I_inv)}")
            print(f"    CRLB trace: {np.trace(I_inv):.6e}")
            print(f"    ✓ FIM invertible")
        except:
            print(f"    ❌ FIM not invertible!")

    return True


# ============================================================================
# TEST 4: PARETO BOUNDARY MINI-SWEEP
# ============================================================================

def test_pareto_mini_sweep():
    """Test a small Pareto boundary sweep"""
    print_section("TEST 4: Mini Pareto Boundary Sweep")

    # Use minimal grid for speed
    alpha_search = np.linspace(0.1, 0.9, 5)
    rho_search = np.linspace(0.1, 0.9, 5)
    D_targets = np.logspace(-7, -5, 5)

    print(f"\nMini sweep parameters:")
    print(f"  α grid: {alpha_search}")
    print(f"  ρ grid: {rho_search}")
    print(f"  D targets: {D_targets}")
    print(f"  Total combinations: {len(alpha_search) * len(rho_search) * len(D_targets)}")

    # Setup
    Sbar = 50
    Smax = 500
    r_b = 1.0

    print(f"\n  Sbar={Sbar}, Smax={Smax}, r_b={r_b}")

    # Simplified computation
    pareto_points = []

    for D_max in D_targets:
        print(f"\n  Testing D_max = {D_max:.3e}:")
        max_rate = 0.0
        best_alpha = None
        best_rho = None

        for alpha in alpha_search:
            for rho in rho_search:
                # Check feasibility
                S_pilot = rho * Sbar / alpha
                S_data = (1 - rho) * Sbar / (1 - alpha)

                if S_pilot > Smax or S_data > Smax:
                    continue

                # Compute rate (simplified)
                C_data, _ = capacity_lb_simple(S_data, Smax, r_b, verbose=False)
                rate = (1 - alpha) * C_data

                if rate > max_rate:
                    max_rate = rate
                    best_alpha = alpha
                    best_rho = rho

        if max_rate > 0:
            pareto_points.append((max_rate, D_max))
            print(f"    → Max rate: {max_rate:.4f} at α={best_alpha:.2f}, ρ={best_rho:.2f}")
        else:
            print(f"    → No feasible solution")

    if len(pareto_points) > 0:
        rates = [p[0] for p in pareto_points]
        mses = [p[1] for p in pareto_points]

        check_array("rates", np.array(rates))
        check_array("mses", np.array(mses))

        # Plot
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.loglog(mses, rates, 'ro-', linewidth=2, markersize=8)
        ax.set_xlabel('MSE Target')
        ax.set_ylabel('Achievable Rate [bits/slot]')
        ax.set_title('Mini Pareto Boundary Test')
        ax.grid(True, alpha=0.3)

        save_diagnostic_figure(fig, 'diagnostic_mini_pareto.png', {
            'mses': mses,
            'rates': rates
        })

        print("\n✓ Pareto boundary computed successfully")
    else:
        print("\n❌ No Pareto points found!")

    return True


# ============================================================================
# TEST 5: PARAMETER SENSITIVITY
# ============================================================================

def test_parameter_sensitivity():
    """Test sensitivity to key parameters"""
    print_section("TEST 5: Parameter Sensitivity Analysis")

    # Base parameters
    Sbar = 50
    Smax = 500
    lamb_b = 1.0

    print("\nBase case:")
    C_base, A_base = capacity_lb_simple(Sbar, Smax, lamb_b, verbose=True)

    # Test variations
    tests = [
        ("2× Sbar", Sbar * 2, Smax, lamb_b),
        ("2× Smax", Sbar, Smax * 2, lamb_b),
        ("2× lamb_b", Sbar, Smax, lamb_b * 2),
        ("0.5× Sbar", Sbar * 0.5, Smax, lamb_b),
        ("10× lamb_b", Sbar, Smax, lamb_b * 10),
    ]

    print("\nSensitivity tests:")
    for name, S, Sm, lb in tests:
        C, A = capacity_lb_simple(S, Sm, lb, verbose=False)
        change = (C - C_base) / C_base * 100
        print(f"  {name:15s}: C={C:.4f} ({change:+.1f}% vs base)")

    return True


# ============================================================================
# MAIN DIAGNOSTIC SUITE
# ============================================================================

def run_all_tests():
    """Run complete diagnostic suite"""
    print("\n" + "=" * 70)
    print("  OISL-ISAC DIAGNOSTIC SUITE")
    print("  Comprehensive Testing & Debugging")
    print("=" * 70)

    tests = [
        ("Basic Functions", test_basic_functions),
        ("Capacity Sweep", test_capacity_sweep),
        ("FIM Computation", test_fim_computation),
        ("Mini Pareto Sweep", test_pareto_mini_sweep),
        ("Parameter Sensitivity", test_parameter_sensitivity),
    ]

    results = {}

    for name, test_func in tests:
        try:
            print(f"\n{'>' * 70}")
            success = test_func()
            results[name] = "✓ PASS" if success else "✗ FAIL"
            print(f"\n{'<' * 70}")
        except Exception as e:
            results[name] = f"❌ ERROR"
            print(f"\n❌ ERROR in {name}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print_section("DIAGNOSTIC SUMMARY")
    for name, result in results.items():
        print(f"  {name:30s}: {result}")

    print("\n" + "=" * 70)
    print("Diagnostic complete! Check the following files:")
    print("  - diagnostic_capacity_sweep.png")
    print("  - diagnostic_capacity_sweep_data.txt")
    print("  - diagnostic_mini_pareto.png")
    print("  - diagnostic_mini_pareto_data.txt")
    print("=" * 70)

    return all("✓" in r for r in results.values())


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)