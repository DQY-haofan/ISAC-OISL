#!/usr/bin/env python3
"""
快速测试gap问题
"""

import numpy as np
import sys

sys.path.append('.')
from isac_core import capacity_lb, capacity_ub_dual

print("=" * 70)
print("🧪 快速Gap测试")
print("=" * 70)

# 测试原始代码会产生负gap的区域
lambda_b_range = np.logspace(-2, 2, 50)
S_bar = 0.5
S_max = 100
dt = 1e-6
tau_d = 50e-9
M_pixels = 16

# 计算有效峰值
S_max_eff = min(S_max, M_pixels * dt / tau_d)

print(f"\n测试参数:")
print(f"  S̄ = {S_bar}")
print(f"  S_max_eff = {S_max_eff:.2f}")
print(f"  λ_b范围: [{lambda_b_range[0]:.2e}, {lambda_b_range[-1]:.2e}]")

print(f"\n计算中...")

negative_gaps = []
zero_gaps = []
positive_gaps = []

for i, lambda_b in enumerate(lambda_b_range):
    try:
        # ⭐ 注意：这里用 S_max_eff 而不是 S_max
        C_lb, _ = capacity_lb(S_bar, S_max_eff, lambda_b, dt, tau_d, M_pixels)
        C_ub, _, _ = capacity_ub_dual(S_bar, S_max_eff, lambda_b, dt, tau_d, M_pixels)

        gap = C_ub - C_lb

        if gap < -1e-6:
            negative_gaps.append((i, lambda_b, C_lb, C_ub, gap))
        elif abs(gap) < 1e-10:
            zero_gaps.append((i, lambda_b, C_lb, C_ub))
        else:
            positive_gaps.append((i, lambda_b, C_lb, C_ub, gap))
    except Exception as e:
        print(f"  ⚠️ 点{i} (λ_b={lambda_b:.2e}) 计算失败: {e}")

# 统计
print(f"\n" + "=" * 70)
print(f"📊 结果统计")
print(f"=" * 70)
print(f"总点数: {len(lambda_b_range)}")
print(f"  负gap: {len(negative_gaps)} ({'❌' if len(negative_gaps) > 0 else '✅'})")
print(f"  零gap: {len(zero_gaps)}")
print(f"  正gap: {len(positive_gaps)}")

if negative_gaps:
    print(f"\n⚠️ 负Gap详情（前5个）:")
    print(f"{'Index':<8} {'λ_b':<12} {'C_LB':<12} {'C_UB':<12} {'Gap':<12}")
    print("-" * 70)
    for item in negative_gaps[:5]:
        idx, lb, c_lb, c_ub, gap = item
        print(f"{idx:<8} {lb:<12.4e} {c_lb:<12.6f} {c_ub:<12.6f} {gap:<12.6f}")

    if len(negative_gaps) > 5:
        print(f"  ... 还有 {len(negative_gaps) - 5} 个")

    # 分析负gap的分布
    lambda_b_negative = [x[1] for x in negative_gaps]
    print(f"\n  λ_b范围: [{min(lambda_b_negative):.2e}, {max(lambda_b_negative):.2e}]")

    # 检查是否集中在某个区域
    low_bg_count = sum(1 for lb in lambda_b_negative if lb < 0.1)
    high_bg_count = sum(1 for lb in lambda_b_negative if lb > 10)

    print(f"\n  分布:")
    print(f"    低背景 (λ_b<0.1): {low_bg_count}/{len(negative_gaps)}")
    print(f"    高背景 (λ_b>10):  {high_bg_count}/{len(negative_gaps)}")

print("\n" + "=" * 70)
print("💡 建议:")
print("=" * 70)

if len(negative_gaps) == 0:
    print("✅ 太好了！没有负gap，修复成功！")
elif len(negative_gaps) < 5:
    print("⚠️ 只有少量负gap，可以在主程序中强制修正:")
    print("   在run_section_iv.py中添加：")
    print("   capacities_ub = np.maximum(capacities_ub, capacities_lb)")
else:
    print("❌ 仍有较多负gap，需要改进capacity_ub_dual的搜索策略")
    print("\n请检查:")
    print("  1. capacity_ub_dual的搜索网格是否够密集？")
    print("  2. lambda_q和nu的范围是否合理？")
    print("  3. 是否需要增加K_max？")

print("=" * 70)