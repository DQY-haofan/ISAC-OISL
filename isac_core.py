#!/usr/bin/env python3
"""
OISL-ISAC 核心函数 - 理论正确修复版
====================================

修复内容：
1. 添加真正的对偶上界（Dual Upper Bound）
2. 物理背景模型接受 dt_slot 参数，单位正确
3. AB 算法正确标注为离散输入容量（下界）
"""

import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar, minimize
import warnings
import time


# [硬件配置部分保持不变，使用前面修复的版本]
# ... (HardwareConfig, AccelerationStrategy 等)

# 为了完整性，这里包含简化版本
class HardwareConfig:
    def __init__(self):
        self.numba_available = False
        self.gpu_available = False
        self.xp = np
        self.jit = lambda *args, **kwargs: lambda f: f
        self.prange = range

        try:
            from numba import jit, prange
            self.numba_available = True
            self.jit = jit
            self.prange = prange
        except:
            pass

        try:
            import cupy as cp
            test = cp.array([1.0])
            _ = test + 1
            self.gpu_available = True
            self.cp = cp
        except:
            self.cp = np

        self._print_config()

    def _print_config(self):
        print("\n" + "=" * 60)
        print("🔧 OISL-ISAC 硬件配置")
        print("=" * 60)
        print(f"NumPy: {np.__version__}")

        if self.numba_available:
            from numba import __version__ as numba_version
            print(f"Numba: {numba_version} ✅ (CPU加速已启用)")
        else:
            print(f"Numba: 未安装")

        if self.gpu_available:
            print(f"CuPy: {self.cp.__version__} ✅")
            try:
                device = self.cp.cuda.Device()
                # 兼容 CuPy 13.x
                try:
                    props = self.cp.cuda.runtime.getDeviceProperties(device.id)
                    gpu_name = props['name'].decode() if isinstance(props['name'], bytes) else str(props['name'])
                except:
                    gpu_name = f"GPU Device {device.id}"
                print(f"GPU: {gpu_name}")
            except:
                pass
        else:
            print(f"CuPy: 未安装")

        print("=" * 60 + "\n")


_hw_config = HardwareConfig()


class AccelerationStrategy:
    NUMBA_THRESHOLD = 10
    GPU_THRESHOLD = 50

    @staticmethod
    def select_method(task_name, data_size):
        if data_size > AccelerationStrategy.GPU_THRESHOLD and _hw_config.gpu_available:
            return 'gpu'
        elif data_size > AccelerationStrategy.NUMBA_THRESHOLD and _hw_config.numba_available:
            return 'numba'
        else:
            return 'cpu'


# ============================================================================
# 容量下界计算（保持不变）
# ============================================================================

def capacity_lb(S_bar, S_max, lambda_b, dt=1e-6, tau_d=50e-9, M_pixels=16):
    """
    容量下界（二元 ON-OFF 输入）

    这是理论下界，对应 Proposition 2
    """

    S_max_eff = S_max
    if tau_d > 0 and M_pixels > 0:
        S_max_eff = min(S_max, M_pixels * dt / tau_d)

    A_min = S_bar
    A_max = S_max_eff
    N_grid = 100

    A_vals = np.linspace(A_min, A_max, N_grid)
    K_max = int(np.ceil(lambda_b + S_max_eff + 10 * np.sqrt(lambda_b + S_max_eff)))
    K_max = min(K_max, 500)

    C_best = 0.0
    A_opt = S_bar

    for A in A_vals:
        p = S_bar / A
        if p > 1.0:
            continue

        I = _mutual_information_binary_cpu(A, p, lambda_b, K_max)

        if I > C_best:
            C_best = I
            A_opt = A

    return C_best, A_opt


def _mutual_information_binary_cpu(A, p, lambda_b, K_max):
    """二元输入互信息（CPU版本）"""
    k_vals = np.arange(K_max)

    log_P0 = -lambda_b + k_vals * np.log(lambda_b + 1e-20) - gammaln(k_vals + 1)
    P0 = np.exp(log_P0)
    P0 = P0 / (np.sum(P0) + 1e-20)

    log_PA = -(lambda_b + A) + k_vals * np.log(lambda_b + A + 1e-20) - gammaln(k_vals + 1)
    PA = np.exp(log_PA)
    PA = PA / (np.sum(PA) + 1e-20)

    PY = (1 - p) * P0 + p * PA
    PY = PY / (np.sum(PY) + 1e-20)

    log2 = np.log(2)
    H_Y = -np.sum(np.where(PY > 1e-20, PY * np.log(PY) / log2, 0))
    H_Y0 = -np.sum(np.where(P0 > 1e-20, P0 * np.log(P0) / log2, 0))
    H_YA = -np.sum(np.where(PA > 1e-20, PA * np.log(PA) / log2, 0))

    return H_Y - (1 - p) * H_Y0 - p * H_YA


# ============================================================================
# 容量上界（对偶公式 - Theorem 2）⭐ 新增
# ============================================================================

def capacity_ub_dual(S_bar, S_max_eff, lambda_b, dt=1e-6,
                     tau_d=50e-9, M_pixels=16,
                     lambda_q_range=None, nu_range=None):
    """
    容量对偶上界（Theorem 2）

    C ≤ inf_{Q,ν≥0} { ν·S̄ + sup_{A∈[0,S_max]} [D(Pois(λ_b+A) || Q) - ν·A] }

    参数：
        S_bar: 平均功率约束
        S_max_eff: 峰值功率约束
        lambda_b: 背景光子数
        lambda_q_range: 测试信道参数搜索范围（若为 None 则自动生成）
        nu_range: Lagrange 乘子搜索范围（若为 None 则自动生成）

    返回：
        C_UB: 容量上界
        (lambda_q_opt, nu_opt): 最优参数
        diagnostics: 诊断信息
    """

    # 自动生成搜索范围
    if lambda_q_range is None:
        # 测试信道应覆盖 [λ_b, λ_b + S_max]
        lambda_q_range = np.linspace(lambda_b, lambda_b + S_max_eff, 30)

    if nu_range is None:
        # Lagrange 乘子典型范围
        nu_range = np.logspace(-3, 1, 25)

    # Poisson 截断
    K_max = int(np.ceil(lambda_b + S_max_eff + 12 * np.sqrt(lambda_b + S_max_eff)))
    K_max = min(K_max, 400)
    k_vals = np.arange(K_max)

    C_UB = np.inf
    lambda_q_opt = lambda_b
    nu_opt = 0

    # 2D 搜索：(λ_q, ν)
    for lambda_q in lambda_q_range:
        # 预计算测试信道 Q = Pois(λ_q)
        log_Q = -lambda_q + k_vals * np.log(lambda_q + 1e-20) - gammaln(k_vals + 1)
        Q = np.exp(log_Q)
        Q = Q / (Q.sum() + 1e-20)

        for nu in nu_range:
            # 内层优化：max_A [D(Pois(λ_b+A) || Q) - ν·A]
            max_val = -np.inf

            # 在 A 上进行 1D 搜索
            A_search = np.linspace(0, S_max_eff, 50)

            for A in A_search:
                lambda_total = lambda_b + A

                # 计算 P = Pois(λ_b + A)
                log_P = -lambda_total + k_vals * np.log(lambda_total + 1e-20) - gammaln(k_vals + 1)
                P = np.exp(log_P)
                P = P / (P.sum() + 1e-20)

                # KL 散度 D(P || Q)
                kl_div = 0.0
                for k in range(K_max):
                    if P[k] > 1e-20 and Q[k] > 1e-20:
                        kl_div += P[k] * np.log(P[k] / Q[k])

                # 目标函数值
                val = kl_div - nu * A

                if val > max_val:
                    max_val = val

            # 对偶目标函数
            dual_obj = nu * S_bar + max_val

            if dual_obj < C_UB:
                C_UB = dual_obj
                lambda_q_opt = lambda_q
                nu_opt = nu

    # 转换为 bits/slot
    C_UB = C_UB / np.log(2)

    diagnostics = {
        'lambda_q_opt': lambda_q_opt,
        'nu_opt': nu_opt,
        'method': 'dual_2d_grid'
    }

    return C_UB, (lambda_q_opt, nu_opt), diagnostics


# ============================================================================
# 离散输入容量（AB 算法）⭐ 重新标注
# ============================================================================

def capacity_discrete_input(S_bar, S_max_eff, lambda_b, dt=1e-6,
                            tau_d=50e-9, M_pixels=16,
                            A_grid=None, max_iter=500, tol=1e-5):
    """
    离散输入信道容量（Arimoto-Blahut 算法）

    ⚠️  注意：这是"离散幅度网格上的容量"，对连续幅度信道来说是下界！

    不应标注为"上界"。正确的上界请使用 capacity_ub_dual()。

    返回：
        C_discrete: 离散输入容量
        p_opt: 最优输入分布
        diagnostics: 收敛信息
    """

    if A_grid is None:
        A_grid = np.concatenate([
            np.linspace(0, S_bar, 15),
            np.linspace(S_bar, S_max_eff, 25)
        ])
        A_grid = np.unique(A_grid)

    N_A = len(A_grid)

    if tau_d > 0:
        r_b = lambda_b / dt
        correction_factor = 1.0 / (1 + r_b * tau_d)
        lambda_b_eff = lambda_b * correction_factor
    else:
        lambda_b_eff = lambda_b

    K_max = int(np.ceil(lambda_b_eff + S_max_eff + 12 * np.sqrt(lambda_b_eff + S_max_eff)))
    K_max = min(K_max, 300)

    # AB 算法主体
    return _arimoto_blahut_cpu(A_grid, lambda_b_eff, S_bar, K_max, max_iter, tol)


def _arimoto_blahut_cpu(A_grid, lambda_b, S_bar, K_max, max_iter, tol):
    """Arimoto-Blahut 算法（CPU）"""
    N_A = len(A_grid)

    P_Y_given_A = np.zeros((N_A, K_max))
    k_vals = np.arange(K_max)

    for i, A in enumerate(A_grid):
        lambda_total = lambda_b + A
        if lambda_total > 0:
            log_probs = -lambda_total + k_vals * np.log(lambda_total + 1e-20) - gammaln(k_vals + 1)
            P_Y_given_A[i, :] = np.exp(log_probs)
            P_Y_given_A[i, :] /= (np.sum(P_Y_given_A[i, :]) + 1e-20)

    p_A = np.ones(N_A) / N_A
    I_history = []

    for iteration in range(max_iter):
        P_Y = np.dot(p_A, P_Y_given_A)
        P_Y /= (np.sum(P_Y) + 1e-20)

        I_current = 0.0
        for i in range(N_A):
            if p_A[i] > 1e-20:
                for k in range(K_max):
                    if P_Y_given_A[i, k] > 1e-20 and P_Y[k] > 1e-20:
                        I_current += p_A[i] * P_Y_given_A[i, k] * np.log2(P_Y_given_A[i, k] / P_Y[k])

        I_history.append(I_current)

        if iteration > 0 and abs(I_history[-1] - I_history[-2]) < tol:
            break

        log_weights = np.zeros(N_A)
        for i in range(N_A):
            kl_div = 0.0
            for k in range(K_max):
                if P_Y_given_A[i, k] > 1e-20 and P_Y[k] > 1e-20:
                    kl_div += P_Y_given_A[i, k] * np.log(P_Y_given_A[i, k] / P_Y[k])
            log_weights[i] = kl_div

        def constraint_error(nu):
            weights = np.exp(log_weights - nu * A_grid)
            p_new = weights / (np.sum(weights) + 1e-20)
            return abs(np.dot(p_new, A_grid) - S_bar)

        try:
            result = minimize_scalar(constraint_error, bounds=(0, 20), method='bounded')
            nu_opt = result.x
        except:
            nu_opt = 1.0

        weights = np.exp(log_weights - nu_opt * A_grid)
        p_A = weights / (np.sum(weights) + 1e-20)

    C_discrete = I_history[-1] if I_history else 0.0

    diagnostics = {
        'iterations': len(I_history),
        'converged': len(I_history) < max_iter,
        'I_history': I_history,
        'method': 'arimoto_blahut'
    }

    return C_discrete, p_A, diagnostics


# 向后兼容别名（但不推荐使用）
def capacity_ub_discrete(*args, **kwargs):
    """
    ⚠️  已废弃：这实际上是离散输入容量（下界），不是上界

    请改用：
    - capacity_ub_dual() - 真正的对偶上界
    - capacity_discrete_input() - 离散输入容量（正确命名）
    """
    warnings.warn(
        "capacity_ub_discrete() 实际计算的是离散输入容量（下界），不是上界。"
        "请改用 capacity_ub_dual() 获取真正的上界，或使用 capacity_discrete_input() 明确标注。",
        DeprecationWarning
    )
    return capacity_discrete_input(*args, **kwargs)


# ============================================================================
# 物理背景模型（修复单位）⭐ 关键修复
# ============================================================================

def physical_background_model(sun_angle_deg, fov_urad,
                              orbit_params=None,
                              wavelength=1550e-9,
                              dt_slot=2e-6):  # ⭐ 新增参数：当前时隙持续时间
    """
    完整物理背景模型（单位修复版）

    ⚠️  关键修复：输出按 dt_slot 计算，而非固定 1ms

    组成：
        1. 太阳杂散光（PST 模型）
        2. 地球照（轨道几何）
        3. 黄道光（日心基线）

    参数：
        dt_slot: 当前时隙持续时间 [秒]
                 默认 2µs，应与仿真的 dt 一致！

    返回：
        lambda_b: 总背景光子数 [photons/slot]
                  ⚠️  单位是"当前时隙"的光子数，不是 1ms 的
        components: 各分量字典（单位相同）
    """

    # 太阳杂散光
    sun_angle_deg = np.clip(sun_angle_deg, 10, 180)
    log_pst = -4.0 - 3.0 * np.log10(sun_angle_deg / 10.0)
    log_pst = np.clip(log_pst, -10, -4)
    pst = 10 ** log_pst

    fov_rad = fov_urad * 1e-6
    omega_fov = np.pi * (fov_rad / 2) ** 2

    # ⭐ 关键：先计算"率"（photons/s），最后乘 dt_slot
    solar_flux_baseline = 1e10  # photons/(m²·s·sr) at 1 AU
    A_eff = 1e-4  # 1 cm² 有效口径

    # 太阳杂散光率 [photons/s]
    lambda_solar_rate = pst * solar_flux_baseline * A_eff * omega_fov

    # 地球照
    if orbit_params is not None:
        altitude_km = orbit_params.get('altitude_km', 600)
        earth_phase = orbit_params.get('earth_phase_angle_deg', 90)

        R_earth = 6371
        theta_earth = np.arctan(R_earth / altitude_km)
        omega_earth = 2 * np.pi * (1 - np.cos(theta_earth))

        albedo = 0.3
        phase_factor = np.cos(np.radians(earth_phase)) * 0.5 + 0.5

        earth_flux = solar_flux_baseline * albedo * phase_factor
        lambda_earthshine_rate = earth_flux * A_eff * omega_fov * (omega_earth / (4 * np.pi))
    else:
        # 默认：地球照约为太阳杂散光的一半
        lambda_earthshine_rate = 0.5 * lambda_solar_rate

    # 黄道光
    zodiacal_baseline_rate = 1e-2 * lambda_solar_rate
    ecliptic_factor = 1.0 + 0.3 * np.cos(np.radians(sun_angle_deg))
    lambda_zodiacal_rate = zodiacal_baseline_rate * ecliptic_factor

    # ⭐ 总率 → 当前时隙光子数
    lambda_b_rate = lambda_solar_rate + lambda_earthshine_rate + lambda_zodiacal_rate
    lambda_b = lambda_b_rate * dt_slot

    components = {
        'solar': lambda_solar_rate * dt_slot,
        'earthshine': lambda_earthshine_rate * dt_slot,
        'zodiacal': lambda_zodiacal_rate * dt_slot,
        'total': lambda_b
    }

    return lambda_b, components


# ============================================================================
# FIM 计算（保持不变）
# ============================================================================

def fim_pilot(alpha, rho, Sbar, N, dt, params, dither_seq,
              tau_d=None, S_pilot_override=None, M_pixels=16):
    """FIM 计算（简化版，实际使用时应包含完整实现）"""

    theta_b = params['theta_b']
    sigma_point_sq = params.get('sigma2', 1e-12)
    r_b = params['r_b']

    if tau_d is None:
        tau_d = params.get('tau_d', 50e-9)

    S_max_eff = params.get('Smax', 100)
    if tau_d > 0:
        S_max_eff = min(S_max_eff, M_pixels * dt / tau_d)

    if S_pilot_override is not None:
        A_pilot = S_pilot_override
    else:
        A_pilot = 0.5 * S_max_eff

    N_pilot = int(min(alpha * N, (rho * Sbar * N) / A_pilot))

    mu_true = np.array([params.get('mu_x', 1e-6), params.get('mu_y', 0.5e-6)])

    # 简化实现（实际应使用完整版本）
    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)
    gamma = 1.0 + a * sigma_point_sq

    I = np.zeros((4, 4))

    for n in range(min(N_pilot, len(dither_seq))):
        d_n = dither_seq[n]
        mu_eff = mu_true + d_n

        L_p = (1.0 / gamma) * np.exp(-b * np.dot(mu_eff, mu_eff) / gamma)

        lambda_n_pre = A_pilot * L_p + r_b * dt
        r_n_pre = lambda_n_pre / dt

        if tau_d > 0:
            g_dead = 1.0 / ((1 + r_n_pre * tau_d) ** 2)
            r_n_post = r_n_pre / (1 + r_n_pre * tau_d)
            lambda_n = r_n_post * dt
        else:
            g_dead = 1.0
            lambda_n = lambda_n_pre

        if lambda_n < 1e-20:
            continue

        base_factor = g_dead * A_pilot * L_p

        grad = np.array([
            base_factor * (-2 * b * mu_eff[0] / gamma),
            base_factor * (-2 * b * mu_eff[1] / gamma),
            base_factor * (-a / gamma + a * b * np.dot(mu_eff, mu_eff) / (gamma ** 2)),
            g_dead * dt
        ])

        I += np.outer(grad, grad) / lambda_n

    return I


# ============================================================================
# 辅助函数
# ============================================================================

def poisson_entropy(lambda_param, K_max=None):
    """Poisson 熵计算"""
    if K_max is None:
        K_max = int(np.ceil(lambda_param + 10 * np.sqrt(lambda_param)))

    K_max = min(K_max, 1000)
    k_vals = np.arange(K_max)

    log_probs = -lambda_param + k_vals * np.log(lambda_param + 1e-20) - gammaln(k_vals + 1)
    probs = np.exp(log_probs)
    probs = probs / (probs.sum() + 1e-20)

    H = -np.sum(np.where(probs > 1e-20, probs * np.log2(probs), 0))

    return H


def setup_ieee_style():
    """设置 IEEE 风格绘图"""
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.size': 10,
        'font.family': 'serif',
        'axes.labelsize': 11,
        'axes.titlesize': 12,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (8, 6),
        'figure.dpi': 100,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
        'lines.linewidth': 1.5,
        'lines.markersize': 6,
        'grid.alpha': 0.3,
        'axes.grid': True
    })

    colors = {
        'zodiacal': '#3498db',
        'earthshine': '#f39c12',
        'stray_light': '#e74c3c',
        'lower_bound': '#2ecc71',
        'upper_bound': '#e67e22'
    }

    return colors


def generate_dither_sequence(N, theta_b):
    """生成抖动序列"""
    return np.random.randn(N, 2) * theta_b * 0.5


SPEED_OF_LIGHT = 299792458


# ============================================================================
# 性能测试（演示上下界）
# ============================================================================

def benchmark_bounds():
    """测试上下界实现"""
    print("\n" + "=" * 60)
    print("⚡ 容量界测试")
    print("=" * 60)

    S_bar = 50
    S_max = 100
    lambda_b = 1.0

    print(f"\n参数：S̄={S_bar}, S_max={S_max}, λ_b={lambda_b}")

    # 下界1：二元输入
    print("\n1️⃣  二元输入下界...")
    start = time.time()
    C_lb_binary, _ = capacity_lb(S_bar, S_max, lambda_b)
    t1 = time.time() - start
    print(f"   C_LB (binary) = {C_lb_binary:.4f} bits/slot")
    print(f"   耗时：{t1 * 1000:.1f} ms")

    # 下界2：离散输入（AB）
    print("\n2️⃣  离散输入容量（AB算法）...")
    start = time.time()
    C_discrete, _, diag = capacity_discrete_input(S_bar, S_max, lambda_b, max_iter=100)
    t2 = time.time() - start
    print(f"   C_discrete (AB) = {C_discrete:.4f} bits/slot")
    print(f"   耗时：{t2 * 1000:.1f} ms")
    print(f"   收敛：{diag['iterations']} 次迭代")

    # 上界：对偶公式
    print("\n3️⃣  对偶上界（Dual UB）...")
    start = time.time()
    C_ub, params_opt, diag_ub = capacity_ub_dual(S_bar, S_max, lambda_b)
    t3 = time.time() - start
    print(f"   C_UB (dual) = {C_ub:.4f} bits/slot")
    print(f"   耗时：{t3 * 1000:.1f} ms")
    print(f"   最优参数：λ_q={params_opt[0]:.2f}, ν={params_opt[1]:.4f}")

    # 显示夹逼
    print("\n📊 容量界总结：")
    print(f"   下界（二元）：   {C_lb_binary:.4f} bits/slot")
    print(f"   下界（离散）：   {C_discrete:.4f} bits/slot")
    print(f"   上界（对偶）：   {C_ub:.4f} bits/slot")
    print(f"   上下界差距：     {C_ub - C_lb_binary:.4f} bits/slot")
    print(f"   相对差距：       {(C_ub - C_lb_binary) / C_lb_binary * 100:.2f}%")

    if C_lb_binary <= C_discrete <= C_ub:
        print("\n✅ 界关系正确：C_LB ≤ C_discrete ≤ C_UB")
    else:
        print("\n⚠️  界关系异常，请检查实现")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark_bounds()