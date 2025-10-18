#!/usr/bin/env python3
"""
OISL-ISAC 核心函数 - GPU加速优化版（A100）
====================================

新增功能：
1. GPU批量计算容量界（充分利用A100）
2. 自动检测并选择最优计算方法
3. 保留CPU版本作为后备
"""

import numpy as np
from scipy.special import gammaln
from scipy.optimize import minimize_scalar, minimize
import warnings
import time


# ============================================================================
# 硬件配置（增强版）
# ============================================================================

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
            self.xp = cp  # ⭐ 默认使用 CuPy
        except:
            self.cp = np
            self.xp = np

        self._print_config()

    def _print_config(self):
        print("\n" + "=" * 60)
        print("🔧 OISL-ISAC 硬件配置（优化版）")
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
                try:
                    props = self.cp.cuda.runtime.getDeviceProperties(device.id)
                    gpu_name = props['name'].decode() if isinstance(props['name'], bytes) else str(props['name'])
                except:
                    gpu_name = f"GPU Device {device.id}"

                mem_info = self.cp.cuda.Device().mem_info
                total_mem_gb = mem_info[1] / 1e9

                print(f"GPU: {gpu_name}")
                print(f"显存: {total_mem_gb:.1f} GB")
                print(f"🚀 GPU加速已启用（批量计算模式）")
            except:
                pass
        else:
            print(f"CuPy: 未安装")

        print("=" * 60 + "\n")


_hw_config = HardwareConfig()


# ============================================================================
# GPU 批量计算：容量下界（⭐ 新增）
# ============================================================================

def capacity_lb_batch_gpu(S_bar, S_max, lambda_b_array, dt=1e-6, tau_d=50e-9, M_pixels=16):
    """
    批量计算容量下界（GPU加速）

    参数：
        S_bar: 平均功率约束
        S_max: 峰值功率约束
        lambda_b_array: 背景光子数数组，shape (N,)
        dt, tau_d, M_pixels: 硬件参数

    返回：
        C_array: 容量数组，shape (N,)
        A_opt_array: 最优幅度数组，shape (N,)
    """

    if not _hw_config.gpu_available:
        # 后备：逐点CPU计算
        C_array = []
        A_opt_array = []
        for lambda_b in lambda_b_array:
            C, A_opt = capacity_lb(S_bar, S_max, lambda_b, dt, tau_d, M_pixels)
            C_array.append(C)
            A_opt_array.append(A_opt)
        return np.array(C_array), np.array(A_opt_array)

    cp = _hw_config.cp

    # 有效峰值功率
    S_max_eff = S_max
    if tau_d > 0 and M_pixels > 0:
        S_max_eff = min(S_max, M_pixels * dt / tau_d)

    # 转移到GPU
    lambda_b_gpu = cp.array(lambda_b_array)  # shape (N,)
    N = len(lambda_b_array)

    # A 网格
    A_min = S_bar
    A_max = S_max_eff
    N_A = 100
    A_vals = cp.linspace(A_min, A_max, N_A)  # shape (N_A,)

    # K 网格
    K_max = int(cp.ceil(cp.max(lambda_b_gpu) + S_max_eff + 10 * cp.sqrt(cp.max(lambda_b_gpu) + S_max_eff)))
    K_max = min(K_max, 500)
    k_vals = cp.arange(K_max)  # shape (K,)

    # 广播计算：lambda_b (N,1,1), A (1,N_A,1), k (1,1,K)
    lambda_b_3d = lambda_b_gpu[:, None, None]  # (N, 1, 1)
    A_3d = A_vals[None, :, None]  # (1, N_A, 1)
    k_3d = k_vals[None, None, :]  # (1, 1, K)

    # 计算 p = S_bar / A
    p_vals = S_bar / A_vals  # (N_A,)
    p_3d = p_vals[None, :, None]  # (1, N_A, 1)

    # 过滤 p > 1 的情况
    valid_mask = (p_3d <= 1.0)  # (1, N_A, 1)

    # P(k | X=0): lambda_b only
    log_P0 = -lambda_b_3d + k_3d * cp.log(lambda_b_3d + 1e-20) - cp.array(gammaln(k_3d.get() + 1))
    P0 = cp.exp(log_P0)
    P0 = P0 / (cp.sum(P0, axis=2, keepdims=True) + 1e-20)  # (N, 1, K)

    # P(k | X=A): lambda_b + A
    lambda_total = lambda_b_3d + A_3d  # (N, N_A, 1)
    log_PA = -lambda_total + k_3d * cp.log(lambda_total + 1e-20) - cp.array(gammaln(k_3d.get() + 1))
    PA = cp.exp(log_PA)
    PA = PA / (cp.sum(PA, axis=2, keepdims=True) + 1e-20)  # (N, N_A, K)

    # P(Y): (1-p)*P0 + p*PA
    PY = (1 - p_3d) * P0 + p_3d * PA  # (N, N_A, K)
    PY = PY / (cp.sum(PY, axis=2, keepdims=True) + 1e-20)

    # 互信息：H(Y) - (1-p)*H(Y|X=0) - p*H(Y|X=A)
    log2 = cp.log(2)

    H_Y = -cp.sum(cp.where(PY > 1e-20, PY * cp.log(PY) / log2, 0), axis=2)  # (N, N_A)
    H_Y0 = -cp.sum(cp.where(P0 > 1e-20, P0 * cp.log(P0) / log2, 0), axis=2)  # (N, 1)
    H_YA = -cp.sum(cp.where(PA > 1e-20, PA * cp.log(PA) / log2, 0), axis=2)  # (N, N_A)

    I = H_Y - (1 - p_vals[None, :]) * H_Y0 - p_vals[None, :] * H_YA  # (N, N_A)

    # 应用 valid_mask
    I = cp.where(valid_mask[:, :, 0], I, -cp.inf)

    # 找到每个 lambda_b 的最优 A
    I_max_idx = cp.argmax(I, axis=1)  # (N,)
    C_array = cp.max(I, axis=1)  # (N,)
    A_opt_array = A_vals[I_max_idx]

    # 转回CPU
    return cp.asnumpy(C_array), cp.asnumpy(A_opt_array)


# ============================================================================
# GPU 批量计算：对偶上界（⭐ 新增）
# ============================================================================

def capacity_ub_dual_batch_gpu(S_bar, S_max, lambda_b_array, dt=1e-6, tau_d=50e-9, M_pixels=16):
    """
    批量计算对偶上界（GPU加速）

    返回：
        C_UB_array: shape (N,)
    """

    if not _hw_config.gpu_available:
        # 后备：逐点CPU计算
        C_UB_array = []
        for lambda_b in lambda_b_array:
            C_UB, _, _ = capacity_ub_dual(S_bar, S_max, lambda_b, dt, tau_d, M_pixels)
            C_UB_array.append(C_UB)
        return np.array(C_UB_array)

    cp = _hw_config.cp

    # 有效峰值功率
    S_max_eff = S_max
    if tau_d > 0 and M_pixels > 0:
        S_max_eff = min(S_max, M_pixels * dt / tau_d)

    lambda_b_gpu = cp.array(lambda_b_array)
    N = len(lambda_b_array)

    # 搜索范围
    lambda_q_range = cp.linspace(cp.min(lambda_b_gpu), cp.max(lambda_b_gpu) + S_max_eff, 30)
    nu_range = cp.logspace(-3, 1, 25)
    A_search = cp.linspace(0, S_max_eff, 50)

    K_max = int(cp.ceil(cp.max(lambda_b_gpu) + S_max_eff + 12 * cp.sqrt(cp.max(lambda_b_gpu) + S_max_eff)))
    K_max = min(K_max, 400)
    k_vals = cp.arange(K_max)

    C_UB_array = cp.full(N, cp.inf)

    # 对每个 lambda_b 进行 2D 搜索（仍需循环，但内部向量化）
    for i in range(N):
        lambda_b_i = lambda_b_gpu[i]
        C_UB_min = cp.inf

        for lambda_q in lambda_q_range:
            # 预计算测试信道 Q
            log_Q = -lambda_q + k_vals * cp.log(lambda_q + 1e-20) - cp.array(gammaln(k_vals.get() + 1))
            Q = cp.exp(log_Q)
            Q = Q / (cp.sum(Q) + 1e-20)

            for nu in nu_range:
                # 内层：对所有 A 向量化计算
                lambda_total = lambda_b_i + A_search  # (50,)

                # P = Pois(lambda_b + A) for all A
                log_P = -lambda_total[:, None] + k_vals[None, :] * cp.log(lambda_total[:, None] + 1e-20) - cp.array(
                    gammaln(k_vals.get() + 1))
                P = cp.exp(log_P)
                P = P / (cp.sum(P, axis=1, keepdims=True) + 1e-20)  # (50, K)

                # KL(P || Q) for all A
                kl_div = cp.sum(cp.where((P > 1e-20) & (Q[None, :] > 1e-20),
                                         P * cp.log(P / Q[None, :]), 0), axis=1)  # (50,)

                # max_A [KL - nu*A]
                obj_vals = kl_div - nu * A_search
                max_val = cp.max(obj_vals)

                # 对偶目标
                dual_obj = nu * S_bar + max_val

                if dual_obj < C_UB_min:
                    C_UB_min = dual_obj

        C_UB_array[i] = C_UB_min / cp.log(2)

    return cp.asnumpy(C_UB_array)


# ============================================================================
# 原有函数（保留，用于单点计算和后备）
# ============================================================================

def capacity_lb(S_bar, S_max, lambda_b, dt=1e-6, tau_d=50e-9, M_pixels=16):
    """容量下界（单点版本）"""
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


def capacity_ub_dual(S_bar, S_max_eff, lambda_b, dt=1e-6,
                     tau_d=50e-9, M_pixels=16,
                     lambda_q_range=None, nu_range=None):
    """对偶上界（单点版本）"""
    if lambda_q_range is None:
        lambda_q_range = np.linspace(lambda_b, lambda_b + S_max_eff, 30)

    if nu_range is None:
        nu_range = np.logspace(-3, 1, 25)

    K_max = int(np.ceil(lambda_b + S_max_eff + 12 * np.sqrt(lambda_b + S_max_eff)))
    K_max = min(K_max, 400)
    k_vals = np.arange(K_max)

    C_UB = np.inf
    lambda_q_opt = lambda_b
    nu_opt = 0

    for lambda_q in lambda_q_range:
        log_Q = -lambda_q + k_vals * np.log(lambda_q + 1e-20) - gammaln(k_vals + 1)
        Q = np.exp(log_Q)
        Q = Q / (Q.sum() + 1e-20)

        for nu in nu_range:
            max_val = -np.inf
            A_search = np.linspace(0, S_max_eff, 50)

            for A in A_search:
                lambda_total = lambda_b + A

                log_P = -lambda_total + k_vals * np.log(lambda_total + 1e-20) - gammaln(k_vals + 1)
                P = np.exp(log_P)
                P = P / (P.sum() + 1e-20)

                kl_div = 0.0
                for k in range(K_max):
                    if P[k] > 1e-20 and Q[k] > 1e-20:
                        kl_div += P[k] * np.log(P[k] / Q[k])

                val = kl_div - nu * A

                if val > max_val:
                    max_val = val

            dual_obj = nu * S_bar + max_val

            if dual_obj < C_UB:
                C_UB = dual_obj
                lambda_q_opt = lambda_q
                nu_opt = nu

    C_UB = C_UB / np.log(2)

    diagnostics = {
        'lambda_q_opt': lambda_q_opt,
        'nu_opt': nu_opt,
        'method': 'dual_2d_grid'
    }

    return C_UB, (lambda_q_opt, nu_opt), diagnostics


# ============================================================================
# 离散输入容量（保留原实现）
# ============================================================================

def capacity_discrete_input(S_bar, S_max_eff, lambda_b, dt=1e-6,
                            tau_d=50e-9, M_pixels=16,
                            A_grid=None, max_iter=500, tol=1e-5):
    """离散输入信道容量（Arimoto-Blahut 算法）"""

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


# ============================================================================
# 物理背景模型（完整版，支持配置）
# ============================================================================

def physical_background_model(sun_angle_deg, fov_urad,
                              orbit_params=None,
                              wavelength=1550e-9,
                              dt_slot=2e-6,
                              config=None):
    """
    完整物理背景模型（修复版）

    ✅ 修复要点：
    1. fov_rad = fov_urad * 1e-6（μrad转rad）
    2. Ω = π*(fov_rad/2)^2（立体角）
    3. ⭐ 地球照使用min(omega_earth, omega_fov)限制
    4. 返回λ_b在10^-2~10^1 photons/slot范围

    参数:
        sun_angle_deg: 太阳避让角 [degrees]
        fov_urad: 接收视场 [microradians]
        orbit_params: 轨道参数字典
        wavelength: 波长 [meters]
        dt_slot: 时隙宽度 [seconds]
        config: 配置字典

    返回:
        lambda_b: 背景光子数 [photons/slot]
        components: 各分量字典
    """

    # 从配置读取参数
    if config is not None and 'physical_model' in config:
        pm = config['physical_model']
        A_eff = pm.get('receiver_aperture', 1e-4)
        Delta_lambda = pm.get('filter_bandwidth', 1e-9)
        tau_optics = pm.get('optical_efficiency', 0.7)
        pst_class = pm.get('pst_class', 'nominal')
        albedo_ocean = pm.get('albedo_ocean', 0.05)
        albedo_land = pm.get('albedo_land', 0.25)
        albedo_cloud = pm.get('albedo_cloud', 0.55)
        cloud_cover = pm.get('global_cloud_cover', 0.6)
        zodiacal_base = pm.get('zodiacal_base_1550nm', 3.5e-9)
    else:
        A_eff = 1e-4
        Delta_lambda = 1e-9
        tau_optics = 0.7
        pst_class = 'nominal'
        albedo_ocean = 0.05
        albedo_land = 0.25
        albedo_cloud = 0.55
        cloud_cover = 0.6
        zodiacal_base = 3.5e-9

    # 物理常数
    SSI_1550nm = 0.233  # W·m⁻²·nm⁻¹
    h = 6.626e-34
    c = 3.0e8
    E_photon = (h * c) / wavelength

    # ⭐ 关键修正：角度单位转换
    fov_rad = fov_urad * 1e-6  # μrad → rad
    omega_fov = np.pi * (fov_rad / 2) ** 2  # 立体角 [steradians]

    # PST函数
    def pst_function(theta_deg, performance='nominal'):
        theta_deg = np.clip(theta_deg, 10, 180)

        if performance == 'high_performance':
            ref_points = {10: 1.0e-8, 30: 5.0e-10, 60: 1.0e-11}
        else:
            ref_points = {10: 5.0e-6, 30: 1.0e-7, 60: 5.0e-9}

        if theta_deg <= 10:
            return ref_points[10]
        elif theta_deg <= 30:
            log_pst = np.interp(theta_deg, [10, 30],
                                [np.log10(ref_points[10]), np.log10(ref_points[30])])
            return 10 ** log_pst
        elif theta_deg <= 60:
            log_pst = np.interp(theta_deg, [30, 60],
                                [np.log10(ref_points[30]), np.log10(ref_points[60])])
            return 10 ** log_pst
        else:
            return ref_points[60] * (60 / theta_deg) ** 3

    # 1. 太阳散射光
    pst = pst_function(sun_angle_deg, performance=pst_class)
    L_stray = (SSI_1550nm * pst) / omega_fov  # W·m⁻²·sr⁻¹·nm⁻¹
    P_solar = L_stray * A_eff * omega_fov * (Delta_lambda * 1e9) * tau_optics  # W
    lambda_solar_rate = P_solar / E_photon  # photons/s

    # 2. 地球照（⭐ 修复版）
    if orbit_params is not None:
        altitude_km = orbit_params.get('altitude_km', 600)
        earth_phase = orbit_params.get('earth_phase_angle_deg', 90)

        R_earth = 6371
        theta_earth = np.arctan(R_earth / altitude_km)
        omega_earth_full = 2 * np.pi * (1 - np.cos(theta_earth))

        # ⭐⭐⭐ 关键修复：地球照应该被接收机视场限制
        omega_earth_visible = min(omega_earth_full, omega_fov)

        alpha_composite = (
                0.71 * (1 - cloud_cover) * albedo_ocean +
                0.29 * (1 - cloud_cover) * albedo_land +
                cloud_cover * albedo_cloud
        )

        phase_factor = np.cos(np.radians(earth_phase)) * 0.5 + 0.5
        L_earthshine = (alpha_composite * SSI_1550nm / np.pi) * phase_factor

        # ⭐ 使用受限的立体角
        P_earth = L_earthshine * A_eff * omega_earth_visible * (Delta_lambda * 1e9) * tau_optics
        lambda_earthshine_rate = P_earth / E_photon
    else:
        # 如果没有轨道参数，使用经验值
        lambda_earthshine_rate = 0.5 * lambda_solar_rate
        L_earthshine = None
        omega_earth_full = None
        omega_earth_visible = None

    # 3. 黄道光
    L_zodiacal_base = zodiacal_base
    ecliptic_factor = 1.0 + 2.4 * (1 - np.cos(np.radians(sun_angle_deg)))
    L_zodiacal = L_zodiacal_base * ecliptic_factor
    P_zodiacal = L_zodiacal * A_eff * omega_fov * (Delta_lambda * 1e9) * tau_optics
    lambda_zodiacal_rate = P_zodiacal / E_photon

    # ⭐ 汇总：rate × dt_slot = photons/slot
    lambda_b_rate = lambda_solar_rate + lambda_earthshine_rate + lambda_zodiacal_rate
    lambda_b = lambda_b_rate * dt_slot

    components = {
        'solar': lambda_solar_rate * dt_slot,
        'earthshine': lambda_earthshine_rate * dt_slot,
        'zodiacal': lambda_zodiacal_rate * dt_slot,
        'total': lambda_b,
        'pst': pst,
        'pst_class': pst_class,
        'L_stray': L_stray,
        'L_earthshine': L_earthshine,
        'L_zodiacal': L_zodiacal,
        'omega_fov': omega_fov,
        'omega_earth_full': omega_earth_full,
        'omega_earth_visible': omega_earth_visible,
        'fov_rad': fov_rad,
    }

    return lambda_b, components


# ============================================================================
# 辅助函数（保留）
# ============================================================================
def fim_pilot(alpha, rho, Sbar, N, dt, params, dither_seq,
              tau_d=None, A_pilot=None, M_pixels=16):
    """
    Fisher Information Matrix 计算（完全遵循Algorithm 1）

    ✅ 修复要点：
    1. A_pilot必须作为固定参数传入（Assumption A2）
    2. 死区修正 g_dead = (1 + r*tau_d)^(-2)
    3. r_b视为rate (photons/s)
    4. 确保N_pilot计算正确

    参数:
        alpha: 时间分配比例 [0,1]
        rho: 光子分配比例 [0,1]
        Sbar: 平均功率约束 (photons/slot)
        N: 总时隙数
        dt: 时隙宽度 (seconds)
        params: 系统参数字典，必须包含:
            - r_b: 背景率 (photons/s) ⚠️
            - theta_b: 波束发散角 (radians)
            - sigma2: 指向方差 (rad^2)
            - mu_x, mu_y: 指向偏差 (radians)
        dither_seq: 抖动序列 (N_pilot × 2)
        tau_d: 死区时间 (seconds)
        A_pilot: 固定pilot幅度 (photons/slot) ⭐ 必传
        M_pixels: 并行像素数

    返回:
        I: Fisher信息矩阵 (4×4)
    """

    # ============================================================================
    # Step 0: 参数提取与验证
    # ============================================================================

    theta_b = params['theta_b']
    sigma_point_sq = params.get('sigma2', 1e-12)
    r_b = params['r_b']  # ⚠️ 这是rate (photons/s)

    if tau_d is None:
        tau_d = params.get('tau_d', 50e-9)

    # ⭐ 计算有效峰值功率（考虑死区）
    S_max_eff = params.get('Smax', 100)
    if tau_d > 0 and M_pixels > 0:
        S_max_eff = min(S_max_eff, M_pixels * dt / tau_d)

    # ⭐ A_pilot必须传入（Assumption A2）
    if A_pilot is None:
        # 后备：使用0.5 S_max_eff
        A_pilot = 0.5 * S_max_eff
        print(f"⚠️ Warning: A_pilot not provided, using default {A_pilot:.2f}")

    # Ensure A_pilot不超过峰值
    A_pilot = min(A_pilot, S_max_eff)

    # ============================================================================
    # Step 1: 计算pilot槽数（Algorithm 1的关键）
    # ============================================================================

    # N_pilot = min{⌊αN⌋, ⌊ρS̄N/A_pilot⌋}
    N_pilot_time = int(alpha * N)
    N_pilot_photon = int((rho * Sbar * N) / A_pilot)
    N_pilot = min(N_pilot_time, N_pilot_photon)

    # 安全检查
    if N_pilot <= 0:
        print(f"⚠️ Warning: N_pilot={N_pilot}, returning zero FIM")
        return np.zeros((4, 4))

    if N_pilot > len(dither_seq):
        print(f"⚠️ Warning: N_pilot={N_pilot} > dither length={len(dither_seq)}")
        N_pilot = len(dither_seq)

    # ============================================================================
    # Step 2: 预计算常数
    # ============================================================================

    mu_true = np.array([params.get('mu_x', 1e-6), params.get('mu_y', 0.5e-6)])

    a = 4.0 / (theta_b ** 2)
    b = 2.0 / (theta_b ** 2)
    gamma = 1.0 + a * sigma_point_sq

    I = np.zeros((4, 4))

    # ============================================================================
    # Step 3: 主循环 - 遍历pilot时隙
    # ============================================================================

    for n in range(N_pilot):
        # --------------------------------------------------------------------
        # Step 3.1: 应用抖动（Proposition 3: Identifiability）
        # --------------------------------------------------------------------
        d_n = dither_seq[n]
        mu_eff = mu_true + d_n

        # --------------------------------------------------------------------
        # Step 3.2: 计算期望指向损耗
        # --------------------------------------------------------------------
        L_p = (1.0 / gamma) * np.exp(-b * np.dot(mu_eff, mu_eff) / gamma)

        # --------------------------------------------------------------------
        # Step 3.3: 计算死区前的光子数（⚠️ 关键单位转换）
        # --------------------------------------------------------------------
        # 信号: A_pilot × L_p (photons/slot)
        # 背景: r_b × dt (photons/s × s = photons/slot)
        lambda_n_pre = A_pilot * L_p + r_b * dt
        r_n_pre = lambda_n_pre / dt  # 转为rate (photons/s)

        # --------------------------------------------------------------------
        # Step 3.4: 死区修正（Proposition 4: Diminishing Returns）
        # --------------------------------------------------------------------
        if tau_d > 0:
            # 非并行死区模型: r' = r / (1 + r*tau_d)
            # 链式法则系数: dr'/dr = 1 / (1 + r*tau_d)^2
            g_dead = 1.0 / ((1 + r_n_pre * tau_d) ** 2)
            r_n_post = r_n_pre / (1 + r_n_pre * tau_d)
            lambda_n = r_n_post * dt
        else:
            g_dead = 1.0
            lambda_n = lambda_n_pre

        # 数值稳定性检查
        if lambda_n < 1e-20:
            continue

        # --------------------------------------------------------------------
        # Step 3.5: 计算偏导数（带链式法则）
        # --------------------------------------------------------------------
        # 基础因子（信号部分对参数的敏感度）
        base_factor = g_dead * A_pilot * L_p

        # ∂λ/∂μx = base_factor × (-2b*μx/γ)
        grad_mux = base_factor * (-2 * b * mu_eff[0] / gamma)

        # ∂λ/∂μy = base_factor × (-2b*μy/γ)
        grad_muy = base_factor * (-2 * b * mu_eff[1] / gamma)

        # ∂λ/∂σ² = base_factor × (-a/γ + ab||μ||²/γ²)
        grad_sigma = base_factor * (
                -a / gamma + a * b * np.dot(mu_eff, mu_eff) / (gamma ** 2)
        )

        # ∂λ/∂r_b = g_dead × dt （背景项的贡献）
        grad_rb = g_dead * dt

        # 梯度向量
        grad = np.array([grad_mux, grad_muy, grad_sigma, grad_rb])

        # --------------------------------------------------------------------
        # Step 3.6: 累积Fisher信息 I += (1/λ) ∇∇ᵀ
        # --------------------------------------------------------------------
        I += np.outer(grad, grad) / lambda_n

    # ============================================================================
    # Step 4: 返回
    # ============================================================================

    return I


# ============================================================================
# 辅助函数：验证FIM条件数
# ============================================================================

def validate_fim(I, J_P=None, threshold=1e30):
    """
    验证FIM的数值稳定性

    参数:
        I: Pilot FIM (4×4)
        J_P: Prior FIM (4×4, optional)
        threshold: 条件数阈值

    返回:
        is_valid: bool
        diagnostics: dict
    """
    J = I.copy()

    if J_P is not None:
        J += J_P

    # 添加正则化
    J += 1e-12 * np.eye(4)

    # 计算条件数
    try:
        cond_num = np.linalg.cond(J)

        if cond_num < threshold:
            J_inv = np.linalg.inv(J)

            # 提取指向参数的CRLB
            W = np.diag([1.0, 1.0, 0.0, 0.0])
            mse_pointing = np.trace(W @ J_inv)

            diagnostics = {
                'valid': True,
                'condition_number': cond_num,
                'mse_pointing': mse_pointing,
                'eigenvalues': np.linalg.eigvals(J).tolist()
            }

            return True, diagnostics
        else:
            diagnostics = {
                'valid': False,
                'condition_number': cond_num,
                'reason': f'Condition number {cond_num:.2e} exceeds threshold'
            }
            return False, diagnostics

    except np.linalg.LinAlgError as e:
        diagnostics = {
            'valid': False,
            'reason': f'LinAlgError: {str(e)}'
        }
        return False, diagnostics


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

if __name__ == "__main__":
    print("=" * 60)
    print("FIM Computation Example (Fixed Version)")
    print("=" * 60)

    # 参数设置
    params = {
        'Sbar': 50.0,
        'Smax': 100.0,
        'dt': 1e-6,
        'N': 10000,
        'theta_b': 10e-6,
        'mu_x': 1e-6,
        'mu_y': 0.5e-6,
        'sigma2': 1e-12,
        'r_b': 1.0,  # ⚠️ photons/s
        'tau_d': 50e-9,
        'M_pixels': 16,
        'J_P': np.diag([1e12, 1e12, 1e6, 1e-3])
    }

    # 资源分配
    alpha = 0.3
    rho = 0.5

    # 计算有效峰值
    Smax_eff = min(params['Smax'],
                   params['M_pixels'] * params['dt'] / params['tau_d'])

    # ⭐ 固定pilot幅度（Assumption A2）
    A_pilot = min(Smax_eff, 4.0 * params['Sbar']) * 0.8

    print(f"\n📊 Configuration:")
    print(f"   S̄ = {params['Sbar']}, S_max_eff = {Smax_eff:.2f}")
    print(f"   α = {alpha}, ρ = {rho}")
    print(f"   A_pilot = {A_pilot:.2f} photons/slot")
    print(f"   r_b = {params['r_b']} photons/s")

    # 生成抖动序列
    N_pilot = int(min(alpha * params['N'],
                      (rho * params['Sbar'] * params['N']) / A_pilot))
    dither_seq = np.random.randn(N_pilot, 2) * params['theta_b'] * 0.5

    print(f"   N_pilot = {N_pilot}")

    # 计算FIM
    print(f"\n🔄 Computing FIM...")
    I_pilot = fim_pilot(
        alpha, rho, params['Sbar'], params['N'],
        params['dt'], params, dither_seq,
        params['tau_d'], A_pilot, params['M_pixels']
    )

    # 验证
    is_valid, diag = validate_fim(I_pilot, params['J_P'])

    print(f"\n✅ Results:")
    print(f"   Valid: {is_valid}")
    print(f"   Condition number: {diag.get('condition_number', 'N/A'):.2e}")
    if is_valid:
        print(f"   MSE(μx,μy): {diag['mse_pointing']:.2e} rad²")
        print(f"   σ(μx,μy): {np.sqrt(diag['mse_pointing']):.2e} rad")

    print(f"\n{'=' * 60}\n")