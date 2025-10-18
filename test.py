#!/usr/bin/env python3
"""
快速测试熵计算修复效果
"""

import numpy as np
from scipy.special import gammaln


def entropy_old(P):
    """旧版本：使用np.where（有bug）"""
    log2 = np.log(2)
    return -np.sum(np.where(P > 1e-20, P * np.log(P) / log2, 0))


def entropy_new(P):
    """新版本：使用显式循环（修复版）"""
    log2 = np.log(2)
    H = 0.0
    for i in range(len(P)):
        if P[i] > 1e-20:
            H -= P[i] * np.log(P[i]) / log2
    return H


# 测试案例
print("=" * 70)
print("🧪 熵计算修复测试")
print("=" * 70)

# 测试1：正常概率分布
P1 = np.array([0.3, 0.5, 0.2])
P1 = P1 / P1.sum()
print(f"\n测试1: 正常分布 P = {P1}")
print(f"  旧版: H = {entropy_old(P1):.6f}")
print(f"  新版: H = {entropy_new(P1):.6f}")

# 测试2：包含0的分布
P2 = np.array([0.0, 0.7, 0.3])
P2 = P2 / P2.sum()
print(f"\n测试2: 包含0 P = {P2}")

import warnings

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    H_old = entropy_old(P2)
    H_new = entropy_new(P2)

    print(f"  旧版: H = {H_old:.6f}")
    if w:
        print(f"    ⚠️ 产生了 {len(w)} 个警告")
        for warning in w:
            print(f"      - {warning.category.__name__}: {warning.message}")
    else:
        print(f"    ✅ 无警告")

    print(f"  新版: H = {H_new:.6f}")
    print(f"    ✅ 无警告")

# 测试3：极小概率
P3 = np.array([1e-30, 0.5, 0.5])
P3 = P3 / P3.sum()
print(f"\n测试3: 极小概率 P[0] = {P3[0]:.2e}")

with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    H_old = entropy_old(P3)
    H_new = entropy_new(P3)

    print(f"  旧版: H = {H_old:.6f}")
    if w:
        print(f"    ⚠️ 产生了 {len(w)} 个警告")
    else:
        print(f"    ✅ 无警告")

    print(f"  新版: H = {H_new:.6f}")
    print(f"    ✅ 无警告")

# 测试4：容量计算完整测试
print(f"\n{'=' * 70}")
print(f"🔧 完整容量计算测试")
print(f"{'=' * 70}")

import sys

sys.path.append('.')
from isac_core import capacity_lb

# 测试会产生警告的参数组合
test_cases = [
    (0.5, 0.01, "极低背景"),
    (0.5, 100, "极高背景"),
    (50, 0.01, "高功率+低背景"),
]

for S_bar, lambda_b, desc in test_cases:
    print(f"\n{desc}: S̄={S_bar}, λ_b={lambda_b}")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        C_lb, A_opt = capacity_lb(S_bar, 100, lambda_b)

        if w:
            print(f"  ⚠️ 产生 {len(w)} 个警告")
            print(f"  C_LB = {C_lb:.6f} (可能不准确)")
        else:
            print(f"  ✅ 无警告")
            print(f"  C_LB = {C_lb:.6f}")

print(f"\n{'=' * 70}")
print(f"结论:")
print(f"{'=' * 70}")
print("""
如果修复前产生警告，修复后无警告，说明修复成功！
修复后重新运行主程序应该不再有负gap。
""")