# !/usr/bin/env python3
"""
快速诊断脚本 - 定位isac_core.py中的问题
运行这个脚本，告诉我输出结果
"""

import numpy as np
import sys

sys.path.append('.')

print("=" * 70)
print("🔍 OISL-ISAC 诊断脚本")
print("=" * 70)

# ============================================================================
# 诊断 1: 检查physical_background_model是否有重复定义
# ============================================================================
print("\n📋 诊断 1: 检查 physical_background_model")
print("-" * 70)

try:
    from isac_core import physical_background_model
    import inspect

    # 获取函数签名
    sig = inspect.signature(physical_background_model)
    print(f"✅ 函数已导入")
    print(f"   参数: {list(sig.parameters.keys())}")

    # 测试调用
    test_config = {
        'physical_model': {
            'receiver_aperture': 1e-4,
            'filter_bandwidth': 1e-9,
            'optical_efficiency': 0.7,
            'pst_class': 'nominal',
            'zodiacal_base_1550nm': 3.5e-9,
            'albedo_ocean': 0.05,
            'albedo_land': 0.25,
            'albedo_cloud': 0.55,
            'global_cloud_cover': 0.6
        }
    }

    orbit_params = {
        'altitude_km': 600,
        'earth_phase_angle_deg': 90
    }

    # 测试3个典型点
    test_points = [
        (30, 50, "近太阳，窄FoV"),
        (90, 100, "垂直，中等FoV"),
        (150, 200, "远太阳，宽FoV")
    ]

    print(f"\n   测试3个典型点（dt=1µs）:")
    for sun_angle, fov_urad, desc in test_points:
        lambda_b, comp = physical_background_model(
            sun_angle, fov_urad,
            orbit_params=orbit_params,
            wavelength=1550e-9,
            dt_slot=1e-6,
            config=test_config
        )

        print(f"\n   {desc}:")
        print(f"     λ_b = {lambda_b:.4e} photons/slot")
        print(f"     Solar = {comp['solar']:.4e}")
        print(f"     Earth = {comp['earthshine']:.4e}")
        print(f"     Zodi  = {comp['zodiacal']:.4e}")

        # 检查是否异常
        if lambda_b > 100:
            print(f"     ⚠️ 异常：λ_b过大！应该在10^-2~10^1范围")
        elif lambda_b < 1e-4:
            print(f"     ⚠️ 异常：λ_b过小！")
        else:
            print(f"     ✅ 正常范围")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# 诊断 2: 检查fim_pilot是否接受A_pilot参数
# ============================================================================
print("\n" + "=" * 70)
print("📋 诊断 2: 检查 fim_pilot")
print("-" * 70)

try:
    from isac_core import fim_pilot, generate_dither_sequence
    import inspect

    sig = inspect.signature(fim_pilot)
    params_list = list(sig.parameters.keys())

    print(f"✅ 函数已导入")
    print(f"   参数: {params_list}")

    if 'A_pilot' in params_list:
        print(f"   ✅ 包含 A_pilot 参数")
    else:
        print(f"   ❌ 缺少 A_pilot 参数！")

    # 测试调用
    params = {
        'Sbar': 50.0,
        'Smax': 100.0,
        'dt': 1e-6,
        'N': 10000,
        'theta_b': 10e-6,
        'mu_x': 1e-6,
        'mu_y': 0.5e-6,
        'sigma2': 1e-12,
        'r_b': 1.0,
        'tau_d': 50e-9,
        'M_pixels': 16,
        'J_P': np.diag([1e12, 1e12, 1e6, 1e-3])
    }

    alpha = 0.3
    rho = 0.5
    N_pilot = int(alpha * params['N'])
    dither_seq = generate_dither_sequence(N_pilot, params['theta_b'])

    # 计算Smax_eff
    Smax_eff = min(params['Smax'],
                   (params['dt'] / params['tau_d']) * params['M_pixels'])
    A_pilot_use = min(Smax_eff, 4.0 * params['Sbar']) * 0.8

    print(f"\n   测试参数:")
    print(f"     Smax_eff = {Smax_eff:.2f}")
    print(f"     A_pilot = {A_pilot_use:.2f}")

    # 尝试调用
    try:
        I_pilot = fim_pilot(
            alpha, rho, params['Sbar'], params['N'],
            params['dt'], params, dither_seq,
            params['tau_d'],
            A_pilot=A_pilot_use,
            M_pixels=params['M_pixels']
        )

        print(f"\n   ✅ fim_pilot 调用成功")
        print(f"     FIM shape: {I_pilot.shape}")
        print(f"     FIM diagonal: {np.diag(I_pilot)}")

        # 检查是否全零
        if np.all(I_pilot == 0):
            print(f"     ❌ FIM全为零！")
        else:
            # 计算2×2子块
            J = I_pilot + params['J_P'] + 1e-12 * np.eye(4)
            J_mu = J[:2, :2]
            cond_num = np.linalg.cond(J_mu)

            print(f"     条件数(2×2子块): {cond_num:.2e}")

            if cond_num < 1e18:
                mse = np.trace(np.linalg.inv(J_mu))
                print(f"     MSE(μx,μy): {mse:.4e} rad²")
                print(f"     ✅ FIM正常工作")
            else:
                print(f"     ⚠️ 条件数过大，矩阵接近奇异")

    except TypeError as e:
        print(f"   ❌ 调用失败: {e}")
        print(f"   可能不支持A_pilot参数")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# 诊断 3: 检查capacity_lb是否正常
# ============================================================================
print("\n" + "=" * 70)
print("📋 诊断 3: 检查 capacity_lb")
print("-" * 70)

try:
    from isac_core import capacity_lb

    S_bar = 50
    S_max = 100
    lambda_b_values = [0.01, 1.0, 10.0]
    dt = 1e-6
    tau_d = 50e-9
    M_pixels = 16

    print(f"✅ 函数已导入")
    print(f"\n   测试3个背景值:")

    for lambda_b in lambda_b_values:
        C_lb, A_opt = capacity_lb(S_bar, S_max, lambda_b, dt, tau_d, M_pixels)

        print(f"\n   λ_b = {lambda_b:.2f} photons/slot:")
        print(f"     C_LB = {C_lb:.4f} bits/slot")
        print(f"     A_opt = {A_opt:.2f}")

        if C_lb < 0 or C_lb > 2:
            print(f"     ⚠️ 容量异常！应该在0-1范围")
        else:
            print(f"     ✅ 容量正常")

except Exception as e:
    print(f"❌ 错误: {e}")
    import traceback

    traceback.print_exc()

# ============================================================================
# 诊断 4: 检查isac_core.py中是否有重复函数定义
# ============================================================================
print("\n" + "=" * 70)
print("📋 诊断 4: 扫描 isac_core.py 源文件")
print("-" * 70)

try:
    with open('isac_core.py', 'r', encoding='utf-8') as f:
        content = f.read()

    # 查找physical_background_model定义
    import re

    pattern = r'^def physical_background_model\('
    matches = list(re.finditer(pattern, content, re.MULTILINE))

    print(f"✅ 文件已读取 ({len(content)} 字符)")
    print(f"\n   找到 'def physical_background_model(' 定义: {len(matches)} 次")

    if len(matches) > 1:
        print(f"   ❌ 警告：有 {len(matches)} 个同名函数定义！")
        print(f"   Python会使用最后一个定义，前面的会被覆盖")
        for i, match in enumerate(matches, 1):
            line_num = content[:match.start()].count('\n') + 1
            print(f"   定义 {i} 在第 {line_num} 行")
    elif len(matches) == 1:
        line_num = content[:matches[0].start()].count('\n') + 1
        print(f"   ✅ 只有1个定义（第 {line_num} 行）")
    else:
        print(f"   ❌ 没有找到定义！")

    # 检查fim_pilot的A_pilot参数
    pattern_fim = r'def fim_pilot\([^)]+\)'
    matches_fim = list(re.finditer(pattern_fim, content, re.MULTILINE))

    if matches_fim:
        print(f"\n   找到 'def fim_pilot(' 定义: {len(matches_fim)} 次")
        for i, match in enumerate(matches_fim, 1):
            line_num = content[:match.start()].count('\n') + 1
            func_def = match.group(0)
            has_a_pilot = 'A_pilot' in func_def or 'S_pilot_override' in func_def
            print(f"   定义 {i} 在第 {line_num} 行")
            print(f"     包含A_pilot参数: {'✅' if has_a_pilot else '❌'}")

except Exception as e:
    print(f"❌ 错误: {e}")

# ============================================================================
# 总结
# ============================================================================
print("\n" + "=" * 70)
print("📊 诊断总结")
print("=" * 70)
print("\n请将以上输出完整复制给我，我会根据结果提供针对性的修复方案。")
print("\n特别注意：")
print("  1. physical_background_model 的 λ_b 值是否在 10^-2 ~ 10^1 范围")
print("  2. fim_pilot 是否支持 A_pilot 参数")
print("  3. isac_core.py 中是否有重复的函数定义")
print("=" * 70)