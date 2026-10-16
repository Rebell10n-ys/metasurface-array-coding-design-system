# compare_user_matrices.py
import os
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'python'))
from metasurface_designer import compute_steering_matrix, calculate_farfield

# 1. 加载三个矩阵
f_fast = PROJECT_ROOT / 'output/fast1.txt'
f_ana = PROJECT_ROOT / 'output/analytical.txt'
f_ga = PROJECT_ROOT / 'output/ga1.txt'

for path in (f_fast, f_ana, f_ga):
    if not path.is_file():
        raise FileNotFoundError(f"请先生成相同物理条件下的待比较矩阵：{path}")

mat_fast = np.loadtxt(f_fast, delimiter=',')
mat_ana = np.loadtxt(f_ana, delimiter=',')
mat_ga = np.loadtxt(f_ga, delimiter=',')
for matrix in (mat_fast, mat_ana, mat_ga):
    if matrix.shape != (36, 36) or not np.isin(matrix, [0, 1, 2, 3]).all():
        raise ValueError("当前比较脚本需要三个 36×36、2-bit 的整数编码矩阵")
mat_fast, mat_ana, mat_ga = (matrix.astype(int) for matrix in (mat_fast, mat_ana, mat_ga))

matrices = {
    "1. 混合优化模式 (fast1.txt)": mat_fast,
    "2. 解析综合模式 (analytical.txt)": mat_ana,
    "3. 遗传算法模式 (ga1.txt)": mat_ga
}

# 物理参数 (默认 1.7 THz, 70 um, 36x36, 2-bit)
fre = 1.7
D_um = 70.0
N = 36
nbit = 2

# 计算目标导向矩阵 (theta=30, phi=150)
theta_target = [30.0]
phi_target = [150.0]
W = compute_steering_matrix(N, theta_target, phi_target, fre_thz=fre, period_um=D_um)

lookup_exp = np.exp(-1j * (np.arange(4) * np.pi / 2))

print("=" * 80)
print("用户生成矩阵的采样远场对比（请核对下方固定频率、周期和目标方向）")
print("=" * 80)

results = {}

for name, mat in matrices.items():
    # 1. 目标方向复场的加权相干和适应度
    exp_mat = lookup_exp[mat]
    E_target = np.sum(exp_mat * W)
    fitness = -np.abs(E_target)
    coherent_field = np.abs(E_target)
    
    # 2. 计算完整 2D 远场
    # 高分辨率角域: theta 0..90 (181点), phi 0..360 (361点)
    ff = calculate_farfield(mat, fre_thz=fre, period_um=D_um, nbit=nbit)
    abs_S = ff['abs_S']
    Smax = ff['Smax']
    
    # 峰值位置
    th_grid = ff['theta_grid']
    ph_grid = ff['phi_grid']
    max_idx = np.unravel_index(np.argmax(abs_S), abs_S.shape)
    peak_th = th_grid[max_idx[0]]
    peak_ph = ph_grid[max_idx[1]]
    
    # 指向误差
    err_th = np.abs(peak_th - theta_target[0])
    err_ph = np.abs((peak_ph - phi_target[0] + 180.0) % 360.0 - 180.0)
    target_th_rad, peak_th_rad = np.deg2rad([theta_target[0], peak_th])
    pointing_cos = (np.cos(target_th_rad) * np.cos(peak_th_rad)
                    + np.sin(target_th_rad) * np.sin(peak_th_rad) * np.cos(np.deg2rad(err_ph)))
    pointing_error = np.rad2deg(np.arccos(np.clip(pointing_cos, -1.0, 1.0)))
    
    # 最大采样峰8度球面锥外的最大幅度比；不等同于第一副瓣或连续角域SLL。
    th_mesh, ph_mesh = np.meshgrid(th_grid, ph_grid, indexing='ij')
    cosine_distance = (np.cos(np.deg2rad(th_mesh)) * np.cos(peak_th_rad)
                       + np.sin(np.deg2rad(th_mesh)) * np.sin(peak_th_rad)
                       * np.cos(np.deg2rad(ph_mesh - peak_ph)))
    ang_dist = np.rad2deg(np.arccos(np.clip(cosine_distance, -1.0, 1.0)))
    sidelobe_mask = ang_dist > 8.0 # 排除主瓣 8 度范围
    
    sidelobe_max = np.max(abs_S[sidelobe_mask])
    sll_db = 20 * np.log10(sidelobe_max / Smax)
    
    # 状态分布
    vals, counts = np.unique(mat, return_counts=True)
    dist_dict = dict(zip(vals, counts))
    
    results[name] = {
        "fitness": fitness,
        "coherent_field": coherent_field,
        "Smax": Smax,
        "peak_theta": peak_th,
        "peak_phi": peak_ph,
        "err_theta": err_th,
        "err_phi": err_ph,
        "pointing_error_deg": pointing_error,
        "sidelobe_max": sidelobe_max,
        "sll_db": sll_db,
        "dist": dist_dict
    }

# 打印对比表格
print(f"{'指标项':<25} | {'1. 混合优化 (fast1)':<18} | {'2. 解析综合 (analytical)':<18} | {'3. 遗传算法 (ga1)':<18}")
print("-" * 88)
print(f"{'目标函数适应度 f (越负越好)':<25} | {results['1. 混合优化模式 (fast1.txt)']['fitness']:<18.2f} | {results['2. 解析综合模式 (analytical.txt)']['fitness']:<18.2f} | {results['3. 遗传算法模式 (ga1.txt)']['fitness']:<18.2f}")
print(f"{'目标方向合成场强 |E|':<25} | {results['1. 混合优化模式 (fast1.txt)']['coherent_field']:<18.2f} | {results['2. 解析综合模式 (analytical.txt)']['coherent_field']:<18.2f} | {results['3. 遗传算法模式 (ga1.txt)']['coherent_field']:<18.2f}")
print(f"{'上半球采样峰值 Smax':<25} | {results['1. 混合优化模式 (fast1.txt)']['Smax']:<18.2f} | {results['2. 解析综合模式 (analytical.txt)']['Smax']:<18.2f} | {results['3. 遗传算法模式 (ga1.txt)']['Smax']:<18.2f}")
print(f"{'实际主瓣指向 (θ, φ)':<25} | ({results['1. 混合优化模式 (fast1.txt)']['peak_theta']:.1f}°, {results['1. 混合优化模式 (fast1.txt)']['peak_phi']:.1f}°)       | ({results['2. 解析综合模式 (analytical.txt)']['peak_theta']:.1f}°, {results['2. 解析综合模式 (analytical.txt)']['peak_phi']:.1f}°)       | ({results['3. 遗传算法模式 (ga1.txt)']['peak_theta']:.1f}°, {results['3. 遗传算法模式 (ga1.txt)']['peak_phi']:.1f}°)      ")
print(f"{'俯仰角指向误差 Δθ':<25} | {results['1. 混合优化模式 (fast1.txt)']['err_theta']:<18.2f}° | {results['2. 解析综合模式 (analytical.txt)']['err_theta']:<18.2f}° | {results['3. 遗传算法模式 (ga1.txt)']['err_theta']:<18.2f}°")
print(f"{'方位角指向误差 Δφ':<25} | {results['1. 混合优化模式 (fast1.txt)']['err_phi']:<18.2f}° | {results['2. 解析综合模式 (analytical.txt)']['err_phi']:<18.2f}° | {results['3. 遗传算法模式 (ga1.txt)']['err_phi']:<18.2f}°")
print(f"{'球面指向误差 (度)':<25} | {results['1. 混合优化模式 (fast1.txt)']['pointing_error_deg']:<18.2f} | {results['2. 解析综合模式 (analytical.txt)']['pointing_error_deg']:<18.2f} | {results['3. 遗传算法模式 (ga1.txt)']['pointing_error_deg']:<18.2f}")
print(f"{'主峰8度锥外幅度比 (dB)':<25} | {results['1. 混合优化模式 (fast1.txt)']['sll_db']:<18.2f} dB | {results['2. 解析综合模式 (analytical.txt)']['sll_db']:<18.2f} dB | {results['3. 遗传算法模式 (ga1.txt)']['sll_db']:<18.2f} dB")
print(f"{'目标归一化场幅 |E|/(M*N)':<25} | {results['1. 混合优化模式 (fast1.txt)']['coherent_field']/mat_fast.size:<18.4f} | {results['2. 解析综合模式 (analytical.txt)']['coherent_field']/mat_ana.size:<18.4f} | {results['3. 遗传算法模式 (ga1.txt)']['coherent_field']/mat_ga.size:<18.4f}")

print("\n编码状态分布 (0:0°, 1:90°, 2:180°, 3:270°):")
for name, res in results.items():
    print(f"  {name}: {res['dist']}")
