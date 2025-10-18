#!/usr/bin/env python3
"""
验证熵计算修复是否生效
"""

import sys
import importlib
import warnings

# 强制重新加载模块
if 'isac_core' in sys.modules:
    del sys.modules['isac_core']

sys.path.insert(0, '.')
import isac_core
from isac_core import capacity_lb

print("=" * 70)
print("🔍 验证修复是否生效")
print("=" * 70)

# 测试1：检查函数源码
print("\n1️⃣ 检查 _mutual_information_binary_cpu 函数...")
import inspect

source = inspect.getsource(isac_core._mutual_information_binary_cpu)

# 查找关键修复标记
if "for k in range(K_max):" in source and "if PY[k] > 1e-20:" in source:
    print("   ✅ 找到修复后的循环代码")
    fix_applied = True
elif "np.where(PY > 1e-20" in source:
    print("   ❌ 仍在使用旧的np.where代码")
    fix_applied = False
else:
    print("   ⚠️ 无法确定是否修复")
    fix_applied = None

# 测试2：实际运行测试
print("\n2️⃣ 实际运行测试（会触发bug的参数）...")

test_params = [
    (0.5, 0.01, "极低背景"),
    (0.5, 100, "极高背景"),
]

all_passed = True

for S_bar, lambda_b, desc in test_params:
    print(f"\n   {desc}: S̄={S_bar}, λ_b={lambda_b}")

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        try:
            C_lb, A_opt = capacity_lb(S_bar, 100, lambda_b)

            if w:
                print(f"      ❌ 产生了 {len(w)} 个警告:")
                for warning in w:
                    print(f"         - {warning.category.__name__}")
                all_passed = False
            else:
                print(f"      ✅ 无警告, C_LB={C_lb:.6f}")
        except Exception as e:
            print(f"      ❌ 出错: {e}")
            all_passed = False

# 最终结论
print("\n" + "=" * 70)
print("📊 验证结果")
print("=" * 70)

if fix_applied and all_passed:
    print("✅ 修复已成功应用！可以重新运行主程序了")
    print("\n下一步:")
    print("  python run_section_iv.py --figure 1")
elif fix_applied is False:
    print("❌ 修复未应用！isac_core.py文件没有被正确修改")
    print("\n请检查:")
    print("  1. 是否修改了正确的isac_core.py文件？")
    print("  2. 修改的位置是否正确？(应该在第310行左右)")
    print("  3. 文件是否保存？")
elif not all_passed:
    print("⚠️ 代码已修改但仍有警告，可能需要重启Python解释器")
    print("\n尝试:")
    print("  1. 清除缓存: rm -rf __pycache__ *.pyc")
    print("  2. 如果在Colab/Jupyter，请重启内核")
    print("  3. 重新运行此脚本")
else:
    print("⚠️ 无法确定状态，请手动检查")

print("=" * 70)