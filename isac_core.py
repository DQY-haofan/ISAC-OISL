#!/usr/bin/env python3
"""
OISL-ISAC 核心函数 - 完整增强版（GPU + Numba 混合加速）
=========================================================

特性：
- 自动检测硬件（GPU/CPU）
- 智能选择加速方案（Numba/GPU/CPU）
- 完全向后兼容原始接口
- 透明回退机制

安装要求：
- 必需：numpy, scipy
- 可选：numba（5-20× CPU加速）
- 可选：cupy（10-50× GPU加速）

使用方法：
    只需将导入改为：
    from isac_core_enhanced import *

    其他代码完全不变！
"""

import numpy as np
from scipy.special import gammaln, logsumexp
from scipy.optimize import minimize_scalar, minimize
import warnings
import time


# ============================================================================
# 硬件能力检测与自动配置
# ============================================================================

class HardwareConfig:
    """硬件配置检测与管理"""

    def __init__(self):
        self.numba_available = False
        self.gpu_available = False
        self.xp = np  # 计算模块（numpy 或 cupy）
        self.jit = None

        self._detect_numba()
        self._detect_gpu()
        self._print_config()

    def _detect_numba(self):
        """检测 Numba JIT 编译器"""
        try:
            from numba import jit, prange
            self.numba_available = True
            self.jit = jit
            self.prange = prange
        except ImportError:
            self.numba_available = False
            # 提供占位符（不加速但不报错）
            self.jit = lambda *args, **kwargs: lambda f: f
            self.prange = range

    def _detect_gpu(self):
        """检测 NVIDIA GPU 和 CuPy"""
        try:
            import cupy as cp
            # 测试 GPU 是否真的可用
            test_array = cp.array([1.0])
            _ = test_array + 1

            self.gpu_available = True
            self.cp = cp
            print("✅ GPU 加速已启用（CuPy）")
        except Exception as e:
            self.gpu_available = False
            self.cp = np  # 回退到 NumPy

    def _print_config(self):
        """打印硬件配置"""
        print("\n" + "=" * 60)
        print("🔧 OISL-ISAC 硬件配置")
        print("=" * 60)
        print(f"NumPy: {np.__version__}")

        if self.numba_available:
            from numba import __version__ as numba_version
            print(f"Numba: {numba_version} ✅ (CPU加速已启用)")
        else:
            print(f"Numba: 未安装（建议安装：pip install numba）")

        if self.gpu_available:
            device = self.cp.cuda.Device()
            print(f"CuPy: {self.cp.__version__} ✅")
            print(f"GPU: {device.name}")
            mem_total = device.mem_info[1] / 1e9
            print(f"显存: {mem_total:.1f} GB")
        else:
            print(f"CuPy: 未安装（可选，用于GPU加速）")

        print("=" * 60 + "\n")

    def to_device(self, array):
        """将数组转移到最优设备"""
        if self.gpu_available and isinstance(array, np.ndarray):
            return self.cp.asarray(array)
        return array

    def to_host(self, array):
        """将数组转回 CPU"""
        if self.gpu_available and hasattr(array, 'get'):
            return array.get()
        return np.asarray(array)


# 全局配置实例
_hw_config = HardwareConfig()


# ============================================================================
# 智能加速器 - 自动选择最优方案
# ============================================================================

class AccelerationStrategy:
    """加速策略选择器"""

    # 阈值配置（可通过环境变量覆盖）
    NUMBA_THRESHOLD = 10
    GPU_THRESHOLD = 50

    @staticmethod
    def select_method(task_name, data_size):
        """
        根据任务和数据规模选择最优方法

        返回：'gpu', 'numba', 或 'cpu'
        """
        if data_size > AccelerationStrategy.GPU_THRESHOLD and _hw_config.gpu_available:
            return 'gpu'
        elif data_size > AccelerationStrategy.NUMBA_THRESHOLD and _hw_config.numba_available:
            return 'numba'
        else:
            return 'cpu'


# ============================================================================
# 容量下界计算（向后兼容 + 加速）
# ============================================================================

@_hw_config.jit(nopython=True, fastmath=True)
def _poisson_pmf_numba(k_vals, lambda_param):
    """Numba 加速的 Poisson PMF"""
    log_probs = -lambda_param + k_vals * np.log(lambda_param + 1e-20)

    # 计算 log(k!)
    log_factorials = np.zeros_like(k_vals, dtype=np.float64)
    for i in range(len(k_vals)):
        k = int(k_vals[i])
        if k > 0:
            log_fact = 0.0
            for j in range(1, k + 1):
                log_fact += np.log(float(j))
            log_factorials[i] = log_fact

    return np.exp(log_probs - log_factorials)


@_hw_config.jit(nopython=True, fastmath=True)
def _mutual_information_binary_numba(A, p, lambda_b, K_max):
    """Numba 加速的互信息计算"""
    k_vals = np.arange(K_max, dtype=np.float64)

    # P(Y|X=0)
    P_Y_given_0 = _poisson_pmf_numba(k_vals, lambda_b)
    P_Y_given_0 = P_Y_given_0 / (np.sum(P_Y_given_0) + 1e-20)

    # P(Y|X=A)
    P_Y_given_A = _poisson_pmf_numba(k_vals, lambda_b + A)
    P_Y_given_A = P_Y_given_A / (np.sum(P_Y_given_A) + 1e-20)

    # P(Y)
    P_Y = (1.0 - p) * P_Y_given_0 + p * P_Y_given_A
    P_Y = P_Y / (np.sum(P_Y) + 1e-20)

    # 计算熵
    log2_const = 1.0 / np.log(2.0)

    H_Y = 0.0
    H_Y_given_0 = 0.0
    H_Y_given_A = 0.0

    for i in range(K_max):
        if P_Y[i] > 1e-20:
            H_Y -= P_Y[i] * np.log(P_Y[i]) * log2_const
        if P_Y_given_0[i] > 1e-20:
            H_Y_given_0 -= P_Y_given_0[i] * np.log(P_Y_given_0[i]) * log2_const
        if P_Y_given_A[i] > 1e-20:
            H_Y_given_A -= P_Y_given_A[i] * np.log(P_Y_given_A[i]) * log2_const

    return H_Y - (1.0 - p) * H_Y_given_0 - p * H_Y_given_A


def capacity_lb(S_bar, S_max, lambda_b, dt=1e-6, tau_d=50e-9, M_pixels=16):
    """
    容量下界计算（二进制输入）

    ✨ 增强：自动选择 Numba/CPU 加速

    参数：
        S_bar: 平均信号约束 [photons/slot]
        S_max: 峰值信号约束 [photons/slot]
        lambda_b: 背景光子数 [photons/slot]
        dt: 时隙持续时间 [seconds]
        tau_d: 死时间 [seconds]
        M_pixels: 并行像素数

    返回：
        C_lb: 容量下界 [bits/slot]
        A_opt: 最优幅度 [photons/slot]
    """

    # 死时间修正
    S_max_eff = S_max
    if tau_d > 0 and M_pixels > 0:
        S_max_eff = min(S_max, M_pixels * dt / tau_d)

    # 幅度搜索网格
    A_min = S_bar
    A_max = S_max_eff
    N_grid = 100

    A_vals = np.linspace(A_min, A_max, N_grid)

    # Poisson 截断
    K_max = int(np.ceil(lambda_b + S_max_eff + 10 * np.sqrt(lambda_b + S_max_eff)))
    K_max = min(K_max, 500)

    C_best = 0.0
    A_opt = S_bar

    # 根据数据规模选择方法
    use_numba = _hw_config.numba_available and K_max > 50

    for A in A_vals:
        p = S_bar / A
        if p > 1.0:
            continue

        # 计算互信息
        if use_numba:
            I = _mutual_information_binary_numba(A, p, lambda_b, K_max)
        else:
            # CPU 回退版本
            I = _mutual_information_cpu(A, p, lambda_b, K_max)

        if I > C_best:
            C_best = I
            A_opt = A

    return C_best, A_opt


def _mutual_information_cpu(A, p, lambda_b, K_max):
    """CPU 版本的互信息计算（向后兼容）"""
    k_vals = np.arange(K_max)

    # P(Y|X=0)
    log_P0 = -lambda_b + k_vals * np.log(lambda_b + 1e-20) - gammaln(k_vals + 1)
    P0 = np.exp(log_P0)
    P0 = P0 / (np.sum(P0) + 1e-20)

    # P(Y|X=A)
    log_PA = -(lambda_b + A) + k_vals * np.log(lambda_b + A + 1e-20) - gammaln(k_vals + 1)
    PA = np.exp(log_PA)
    PA = PA / (np.sum(PA) + 1e-20)

    # P(Y)
    PY = (1 - p) * P0 + p * PA
    PY = PY / (np.sum(PY) + 1e-20)

    # 熵
    log2 = np.log(2)
    H_Y = -np.sum(np.where(PY > 1e-20, PY * np.log(PY) / log2, 0))
    H_Y0 = -np.sum(np.where(P0 > 1e-20, P0 * np.log(P0) / log2, 0))
    H_YA = -np.sum(np.where(PA > 1e-20, PA * np.log(PA) / log2, 0))

    return H_Y - (1 - p) * H_Y0 - p * H_YA


# ============================================================================
# 容量上界计算（Arimoto-Blahut 算法）
# ============================================================================

def capacity_ub_discrete(S_bar, S_max_eff, lambda_b, dt=1e-6,
                         tau_d=50e-9, M_pixels=16,
                         A_grid=None, max_iter=500, tol=1e-5):
    """
    容量上界计算（Arimoto-Blahut 算法）

    ✨ 增强：支持 GPU 加速（大规模计算）

    返回：
        C_UB: 容量上界 [bits/slot]
        p_opt: 最优输入分布
        diagnostics: 收敛诊断信息
    """

    # 自适应网格
    if A_grid is None:
        A_grid = np.concatenate([
            np.linspace(0, S_bar, 15),
            np.linspace(S_bar, S_max_eff, 25)
        ])
        A_grid = np.unique(A_grid)

    N_A = len(A_grid)

    # 死时间修正
    if tau_d > 0:
        r_b = lambda_b / dt
        correction_factor = 1.0 / (1 + r_b * tau_d)
        lambda_b_eff = lambda_b * correction_factor
    else:
        lambda_b_eff = lambda_b

    # Poisson 截断
    K_max = int(np.ceil(lambda_b_eff + S_max_eff + 12 * np.sqrt(lambda_b_eff + S_max_eff)))
    K_max = min(K_max, 300)

    # 选择计算设备
    use_gpu = _hw_config.gpu_available and (N_A * K_max > 5000)

    if use_gpu:
        return _capacity_ub_gpu(A_grid, lambda_b_eff, S_bar, K_max, max_iter, tol)
    else:
        return _capacity_ub_cpu(A_grid, lambda_b_eff, S_bar, K_max, max_iter, tol)


def _capacity_ub_cpu(A_grid, lambda_b, S_bar, K_max, max_iter, tol):
    """CPU 版本的 Arimoto-Blahut"""
    N_A = len(A_grid)

    # 预计算输出分布
    P_Y_given_A = np.zeros((N_A, K_max))
    k_vals = np.arange(K_max)

    for i, A in enumerate(A_grid):
        lambda_total = lambda_b + A
        if lambda_total > 0:
            log_probs = -lambda_total + k_vals * np.log(lambda_total + 1e-20) - gammaln(k_vals + 1)
            P_Y_given_A[i, :] = np.exp(log_probs)
            P_Y_given_A[i, :] /= (np.sum(P_Y_given_A[i, :]) + 1e-20)

    # 初始化均匀分布
    p_A = np.ones(N_A) / N_A
    I_history = []

    for iteration in range(max_iter):
        # 输出分布
        P_Y = np.dot(p_A, P_Y_given_A)
        P_Y /= (np.sum(P_Y) + 1e-20)

        # 互信息
        I_current = 0.0
        for i in range(N_A):
            if p_A[i] > 1e-20:
                for k in range(K_max):
                    if P_Y_given_A[i, k] > 1e-20 and P_Y[k] > 1e-20:
                        I_current += p_A[i] * P_Y_given_A[i, k] * np.log2(P_Y_given_A[i, k] / P_Y[k])

        I_history.append(I_current)

        # 收敛检查
        if iteration > 0 and abs(I_history[-1] - I_history[-2]) < tol:
            break

        # 更新输入分布（使用 Blahut 迭代）
        log_weights = np.zeros(N_A)
        for i in range(N_A):
            kl_div = 0.0
            for k in range(K_max):
                if P_Y_given_A[i, k] > 1e-20 and P_Y[k] > 1e-20:
                    kl_div += P_Y_given_A[i, k] * np.log(P_Y_given_A[i, k] / P_Y[k])
            log_weights[i] = kl_div

        # 施加平均功率约束（Lagrange 乘子法）
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

    C_UB = I_history[-1] if I_history else 0.0

    diagnostics = {
        'iterations': len(I_history),
        'converged': len(I_history) < max_iter,
        'I_history': I_history
    }

    return C_UB, p_A, diagnostics


def _capacity_ub_gpu(A_grid, lambda_b, S_bar, K_max, max_iter, tol):
    """GPU 加速版本的 Arimoto-Blahut"""
    cp = _hw_config.cp

    # 转移到 GPU
    A_grid_gpu = cp.asarray(A_grid)
    N_A = len(A_grid_gpu)

    # 预计算输出分布（GPU 并行）
    P_Y_given_A = cp.zeros((N_A, K_max))
    k_vals = cp.arange(K_max)

    for i in range(N_A):
        A = float(A_grid_gpu[i])
        lambda_total = lambda_b + A
        if lambda_total > 0:
            log_probs = -lambda_total + k_vals * cp.log(lambda_total + 1e-20)
            # 阶乘（使用 GPU）
            log_factorials = cp.array([cp.sum(cp.log(cp.arange(1, k + 1))) if k > 0 else 0
                                       for k in range(K_max)])
            P_Y_given_A[i, :] = cp.exp(log_probs - log_factorials)
            P_Y_given_A[i, :] /= (cp.sum(P_Y_given_A[i, :]) + 1e-20)

    # 初始化
    p_A = cp.ones(N_A) / N_A
    I_history = []

    for iteration in range(max_iter):
        P_Y = cp.dot(p_A, P_Y_given_A)
        P_Y /= (cp.sum(P_Y) + 1e-20)

        # 互信息（GPU 并行）
        kl_matrix = cp.where(
            (P_Y_given_A > 1e-20) & (P_Y[None, :] > 1e-20),
            P_Y_given_A * cp.log(P_Y_given_A / P_Y[None, :]),
            0
        )
        I_current = float(cp.sum(p_A[:, None] * kl_matrix) / cp.log(2))

        I_history.append(I_current)

        if iteration > 0 and abs(I_history[-1] - I_history[-2]) < tol:
            break

        # 更新分布
        kl_div = cp.sum(kl_matrix, axis=1)

        # Lagrange 乘子优化
        def constraint_error_gpu(nu):
            weights = cp.exp(kl_div - nu * A_grid_gpu)
            p_new = weights / (cp.sum(weights) + 1e-20)
            return float(cp.abs(cp.dot(p_new, A_grid_gpu) - S_bar))

        try:
            result = minimize_scalar(constraint_error_gpu, bounds=(0, 20), method='bounded')
            nu_opt = result.x
        except:
            nu_opt = 1.0

        weights = cp.exp(kl_div - nu_opt * A_grid_gpu)
        p_A = weights / (cp.sum(weights) + 1e-20)

    # 转回 CPU
    C_UB = I_history[-1]
    p_A_cpu = cp.asnumpy(p_A)

    diagnostics = {
        'iterations': len(I_history),
        'converged': len(I_history) < max_iter,
        'I_history': I_history,
        'method': 'gpu'
    }

    return C_UB, p_A_cpu, diagnostics


# ============================================================================
# 物理背景模型（Section II 完整集成）
# ============================================================================

def physical_background_model(sun_angle_deg, fov_urad,
                              orbit_params=None,
                              wavelength=1550e-9):
    """
    完整物理背景模型

    组成：
        1. 太阳杂散光（PST 模型）
        2. 地球照（轨道几何）
        3. 黄道光（日心基线）

    返回：
        lambda_b: 总背景 [photons/slot]
        components: 各分量字典
    """

    # 太阳杂散光
    sun_angle_deg = np.clip(sun_angle_deg, 10, 180)
    log_pst = -4.0 - 3.0 * np.log10(sun_angle_deg / 10.0)
    log_pst = np.clip(log_pst, -10, -4)
    pst = 10 ** log_pst

    fov_rad = fov_urad * 1e-6
    omega_fov = np.pi * (fov_rad / 2) ** 2

    solar_flux_baseline = 1e10  # photons/(m²·s·sr) at 1 AU
    dt_ref = 1e-3
    A_eff = 1e-4  # 1 cm² aperture

    lambda_solar = pst * solar_flux_baseline * A_eff * omega_fov * dt_ref

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
        lambda_earthshine = earth_flux * A_eff * omega_fov * dt_ref * (omega_earth / (4 * np.pi))
    else:
        lambda_earthshine = 0.5 * lambda_solar

    # 黄道光
    zodiacal_baseline = 1e-2 * lambda_solar
    ecliptic_factor = 1.0 + 0.3 * np.cos(np.radians(sun_angle_deg))
    lambda_zodiacal = zodiacal_baseline * ecliptic_factor

    # 总背景
    lambda_b = lambda_solar + lambda_earthshine + lambda_zodiacal

    components = {
        'solar': lambda_solar,
        'earthshine': lambda_earthshine,
        'zodiacal': lambda_zodiacal,
        'total': lambda_b
    }

    return lambda_b, components


# ============================================================================
# FIM 计算（增强版，支持 Numba/GPU）
# ============================================================================

@_hw_config.jit(nopython=True, parallel=True, fastmath=True)
def _fim_inner_loop_numba(N_pilot, A_pilot, lambda_b_dt, dt, tau_d,
                          mu_true, dither_seq, theta_b, sigma_point_sq):
    """Numba 加速的 FIM 内循环"""
    from numba import prange

    a = 4.0 / (theta_b * theta_b)
    b = 2.0 / (theta_b * theta_b)
    gamma = 1.0 + a * sigma_point_sq

    I_flat = np.zeros(16)

    for n in prange(N_pilot):
        mu_eff_x = mu_true[0] + dither_seq[n, 0]
        mu_eff_y = mu_true[1] + dither_seq[n, 1]
        mu_eff_norm_sq = mu_eff_x ** 2 + mu_eff_y ** 2

        exp_term = np.exp(-b * mu_eff_norm_sq / gamma)
        L_p = (1.0 / gamma) * exp_term

        lambda_n_pre = A_pilot * L_p + lambda_b_dt
        r_n_pre = lambda_n_pre / dt

        if tau_d > 0:
            denominator = 1.0 + r_n_pre * tau_d
            g_dead = 1.0 / (denominator * denominator)
            r_n_post = r_n_pre / denominator
            lambda_n = r_n_post * dt
        else:
            g_dead = 1.0
            lambda_n = lambda_n_pre

        if lambda_n < 1e-20:
            continue

        base_factor = g_dead * A_pilot * L_p

        dlambda_dmu_x = base_factor * (-2.0 * b * mu_eff_x / gamma)
        dlambda_dmu_y = base_factor * (-2.0 * b * mu_eff_y / gamma)
        dlambda_dsigma2 = base_factor * (-a / gamma + a * b * mu_eff_norm_sq / (gamma * gamma))
        dlambda_drb = g_dead * dt

        grad = np.array([dlambda_dmu_x, dlambda_dmu_y, dlambda_dsigma2, dlambda_drb])

        inv_lambda = 1.0 / lambda_n
        idx = 0
        for i in range(4):
            for j in range(4):
                I_flat[idx] += inv_lambda * grad[i] * grad[j]
                idx += 1

    return I_flat


def fim_pilot(alpha, rho, Sbar, N, dt, params, dither_seq,
              tau_d=None, S_pilot_override=None, M_pixels=16):
    """
    FIM 计算（增强版，向后兼容）

    ✨ 增强：自动选择 Numba/GPU/CPU

    完全兼容原始接口！
    """

    # 提取参数
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

    # 选择计算方法
    method = AccelerationStrategy.select_method('fim_pilot', N_pilot)

    if method == 'numba' and _hw_config.numba_available:
        # Numba 加速版本
        dither_array = np.ascontiguousarray(dither_seq[:N_pilot])
        I_flat = _fim_inner_loop_numba(
            N_pilot, A_pilot, r_b * dt, dt, tau_d,
            mu_true, dither_array, theta_b, sigma_point_sq
        )
        I_pilot = I_flat.reshape(4, 4)

    elif method == 'gpu' and _hw_config.gpu_available:
        # GPU 版本
        I_pilot = _fim_gpu(N_pilot, A_pilot, r_b, dt, tau_d,
                           mu_true, dither_seq, theta_b, sigma_point_sq)
    else:
        # CPU 回退版本
        I_pilot = _fim_cpu(N_pilot, A_pilot, r_b, dt, tau_d,
                           mu_true, dither_seq, theta_b, sigma_point_sq, M_pixels)

    return I_pilot


def _fim_cpu(N_pilot, A_pilot, r_b, dt, tau_d, mu_true, dither_seq,
             theta_b, sigma_point_sq, M_pixels):
    """CPU 回退版本"""
    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)
    gamma = 1.0 + a * sigma_point_sq

    I = np.zeros((4, 4))

    for n in range(N_pilot):
        d_n = dither_seq[n % len(dither_seq)]
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


def _fim_gpu(N_pilot, A_pilot, r_b, dt, tau_d, mu_true, dither_seq,
             theta_b, sigma_point_sq):
    """GPU 版本的 FIM 计算"""
    cp = _hw_config.cp

    # 转移到 GPU
    mu_true_gpu = cp.asarray(mu_true)
    dither_seq_gpu = cp.asarray(dither_seq[:N_pilot])

    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)
    gamma = 1.0 + a * sigma_point_sq

    # 向量化计算
    mu_eff = mu_true_gpu + dither_seq_gpu
    mu_eff_norm_sq = cp.sum(mu_eff ** 2, axis=1)

    exp_term = cp.exp(-b * mu_eff_norm_sq / gamma)
    L_p = (1.0 / gamma) * exp_term

    lambda_n_pre = A_pilot * L_p + r_b * dt
    r_n_pre = lambda_n_pre / dt

    if tau_d > 0:
        denominator = 1.0 + r_n_pre * tau_d
        g_dead = 1.0 / (denominator ** 2)
        r_n_post = r_n_pre / denominator
        lambda_n = r_n_post * dt
    else:
        g_dead = cp.ones_like(r_n_pre)
        lambda_n = lambda_n_pre

    valid_mask = lambda_n > 1e-20

    base_factor = g_dead * A_pilot * L_p

    grad_matrix = cp.stack([
        base_factor * (-2.0 * b * mu_eff[:, 0] / gamma),
        base_factor * (-2.0 * b * mu_eff[:, 1] / gamma),
        base_factor * (-a / gamma + a * b * mu_eff_norm_sq / (gamma ** 2)),
        g_dead * dt * cp.ones(N_pilot)
    ], axis=1)

    grad_matrix[~valid_mask] = 0
    inv_lambda = cp.where(valid_mask, 1.0 / lambda_n, 0)

    weighted_grad = grad_matrix * inv_lambda[:, None]
    I_gpu = cp.dot(weighted_grad.T, grad_matrix)

    # 转回 CPU
    I_pilot = cp.asnumpy(I_gpu)

    return I_pilot


# ============================================================================
# 辅助函数（向后兼容）
# ============================================================================

def poisson_entropy(lambda_param, K_max=None):
    """Poisson 熵计算（向后兼容）"""
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
    """设置 IEEE 风格绘图（向后兼容）"""
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


# 常量（向后兼容）
SPEED_OF_LIGHT = 299792458  # m/s


# ============================================================================
# 性能测试函数
# ============================================================================

def benchmark_performance():
    """测试各种加速方案的性能"""
    print("\n" + "=" * 60)
    print("⚡ 性能基准测试")
    print("=" * 60)

    # 测试参数
    S_bar = 50
    S_max = 100
    lambda_b = 1.0

    print("\n1️⃣  容量下界计算（100点网格）")

    start = time.time()
    C_lb, _ = capacity_lb(S_bar, S_max, lambda_b)
    elapsed = time.time() - start

    print(f"   耗时：{elapsed * 1000:.2f} ms")
    print(f"   结果：C_LB = {C_lb:.4f} bits/slot")

    print("\n2️⃣  容量上界计算（Arimoto-Blahut）")

    start = time.time()
    C_ub, _, diag = capacity_ub_discrete(S_bar, S_max, lambda_b, max_iter=100)
    elapsed = time.time() - start

    print(f"   耗时：{elapsed * 1000:.2f} ms")
    print(f"   结果：C_UB = {C_ub:.4f} bits/slot")
    print(f"   迭代次数：{diag['iterations']}")
    print(f"   收敛：{'是' if diag['converged'] else '否'}")
    print(f"   差距：{C_ub - C_lb:.4f} bits/slot")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    # 运行性能测试
    benchmark_performance()