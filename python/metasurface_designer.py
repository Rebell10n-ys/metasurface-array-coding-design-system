#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
Metasurface Designer (超表面阵列编码优化与设计系统)
=============================================================================
支持 0.1–5 THz 频段、1-bit / 2-bit 离散相位编码、单波束与多波束赋形。
提供三种求解模式：
  1. 'ga_exact'    : 遗传算法搜索 (锦标赛选择、双点交叉、单点变异)
  2. 'analytical'  : 目标相位叠加与最近邻量化
  3. 'fast' (推荐) : 解析初始化 + 增量坐标下降微调

使用 NumPy CPU 计算，基于理想等幅、离散相位的阵列因子模型。
输出整数编码矩阵，支持 TXT、CSV 和 MAT 文件。
"""

import sys
import os
import time
import argparse
import numpy as np

# 物理常量
SPEED_OF_LIGHT = 2.99792458e8  # m/s
EPSILON_0 = 8.854187817e-12    # F/m
MU_0 = 4.0 * np.pi * 1e-7      # H/m

def compute_steering_matrix(N, theta, phi, fre_thz=1.7, period_um=70.0, weights=None, M=None):
    """
    预计算导向相位矩阵 W (Coherent Steering Matrix):
      W(m, n) = sum_{k=1}^K w_k * exp(-1j * psi_k(m, n))
    其中:
      psi_k(m, n) = k0 * D * ( (m - 0.5)*sin(th)*cos(ph) + (n - 0.5)*sin(th)*sin(ph) )
    """
    if M is None:
        M = N
        
    fre_hz = float(fre_thz) * 1e12
    D_m = float(period_um) * 1e-6
    c = 1.0 / np.sqrt(EPSILON_0 * MU_0)
    w = fre_hz * 2.0 * np.pi
    k0 = w / c
    
    theta_rad = np.deg2rad(np.atleast_1d(theta))
    phi_rad = np.deg2rad(np.atleast_1d(phi))
    
    if len(theta_rad) != len(phi_rad):
        raise ValueError(f"theta 长度 ({len(theta_rad)}) 与 phi 长度 ({len(phi_rad)}) 必须一致！")
        
    if weights is None:
        weights = np.ones(len(theta_rad), dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64).ravel()
        if len(weights) != len(theta_rad):
            raise ValueError("波束权重 weights 长度必须与目标波束数量一致！")
            
    m_idx = np.arange(1, M + 1) - 0.5
    n_idx = np.arange(1, N + 1) - 0.5
    X_grid, Y_grid = np.meshgrid(m_idx, n_idx, indexing='ij')
    
    W = np.zeros((M, N), dtype=np.complex128)
    for kk in range(len(theta_rad)):
        u_k = np.sin(theta_rad[kk]) * np.cos(phi_rad[kk])
        v_k = np.sin(theta_rad[kk]) * np.sin(phi_rad[kk])
        psi_k = k0 * D_m * (X_grid * u_k + Y_grid * v_k)
        W += weights[kk] * np.exp(-1j * psi_k)
        
    return W

def evaluate_fitness(B, W, nbit=2):
    """
    计算给定编码矩阵 B 的适应度:
    f = -| sum_{m,n} exp(-1j * fa(m,n)) * W(m,n) |
    最大化目标方向复场的加权相干和幅值，返回其相反数。
    """
    num_states = 2**nbit
    step = np.pi / (2**(nbit - 1))
    lookup_exp = np.exp(-1j * (np.arange(num_states) * step))
    E = np.sum(lookup_exp[B] * W)
    return -np.abs(E)

def design_analytical(N, theta, phi, nbit=2, fre_thz=1.7, period_um=70.0, weights=None, M=None):
    """
    Type B1: 理论共轭相位直接综合法
    """
    if M is None:
        M = N
    t0 = time.perf_counter()
    W = compute_steering_matrix(N, theta, phi, fre_thz=fre_thz, period_um=period_um, weights=weights, M=M)
    
    # 单元系数为 exp(-j*phase)，取 phase=angle(W) 使单元贡献同相。
    target_continuous_phase = np.angle(W) % (2.0 * np.pi)
    num_states = 2**nbit
    step = 2.0 * np.pi / num_states
    
    # 最佳最近邻量化
    B = np.round(target_continuous_phase / step).astype(np.int32) % num_states
    fitness = evaluate_fitness(B, W, nbit=nbit)
    elapsed = time.perf_counter() - t0
    
    return {
        'B': B,
        'fitness': fitness,
        'elapsed_sec': elapsed,
        'method': 'analytical',
        'evals': 1
    }

def design_greedy(N, theta, phi, nbit=2, fre_thz=1.7, period_um=70.0, weights=None,
                  M=None, max_passes=4, initial_B=None, seed=42):
    """
    Type B2: 增量贪心坐标下降优化 (Coordinate Descent / Iterative Conditional Modes)
    """
    if M is None:
        M = N
    t0 = time.perf_counter()
    W = compute_steering_matrix(N, theta, phi, fre_thz=fre_thz, period_um=period_um, weights=weights, M=M)
    
    num_states = 2**nbit
    step = np.pi / (2**(nbit - 1))
    lookup_exp = np.exp(-1j * (np.arange(num_states) * step))
    
    if initial_B is None:
        if seed is not None:
            np.random.seed(seed)
        B = np.random.randint(0, num_states, (M, N), dtype=np.int32)
    else:
        B = initial_B.copy().astype(np.int32)
        
    E_curr = np.sum(lookup_exp[B] * W)
    curr_score = np.abs(E_curr)
    
    total_evals = 1
    coords = [(m, n) for m in range(M) for n in range(N)]
    
    for pass_idx in range(max_passes):
        improved = False
        np.random.shuffle(coords)
        for m, n in coords:
            old_val = B[m, n]
            best_val = old_val
            best_score = curr_score
            best_E = E_curr
            
            W_mn = W[m, n]
            old_exp = lookup_exp[old_val]
            
            for v in range(num_states):
                if v == old_val:
                    continue
                total_evals += 1
                cand_E = E_curr + (lookup_exp[v] - old_exp) * W_mn
                cand_score = np.abs(cand_E)
                if cand_score > best_score:
                    best_score = cand_score
                    best_val = v
                    best_E = cand_E
                    
            if best_val != old_val:
                B[m, n] = best_val
                E_curr = best_E
                curr_score = best_score
                improved = True
                
        if not improved:
            break
            
    elapsed = time.perf_counter() - t0
    return {
        'B': B,
        'fitness': -curr_score,
        'elapsed_sec': elapsed,
        'method': 'greedy',
        'evals': total_evals
    }

def design_hybrid(N, theta, phi, nbit=2, fre_thz=1.7, period_um=70.0, weights=None, M=None, max_passes=3):
    """
    Type B3: 推荐默认方案：解析初始化 + 快速坐标下降精修
    """
    t0 = time.perf_counter()
    init_res = design_analytical(N, theta, phi, nbit=nbit, fre_thz=fre_thz, period_um=period_um, weights=weights, M=M)
    opt_res = design_greedy(N, theta, phi, nbit=nbit, fre_thz=fre_thz, period_um=period_um, weights=weights,
                            M=M, max_passes=max_passes, initial_B=init_res['B'])
    elapsed = time.perf_counter() - t0
    return {
        'B': opt_res['B'],
        'fitness': opt_res['fitness'],
        'elapsed_sec': elapsed,
        'method': 'hybrid',
        'evals': init_res['evals'] + opt_res['evals']
    }

def design_ga_exact(N, theta, phi, nbit=2, fre_thz=1.7, period_um=70.0, weights=None, M=None,
                    nPop=50, maxIt=4000, nPc=0.9, nMu=0.8, seed=42):
    """
    使用锦标赛选择、双点交叉、单点变异和精英筛选的遗传算法。
    """
    if M is None:
        M = N
    t0 = time.perf_counter()
    if seed is not None:
        np.random.seed(seed)
        
    W = compute_steering_matrix(N, theta, phi, fre_thz=fre_thz, period_um=period_um, weights=weights, M=M)
    nC = int(round(nPop * nPc / 2.0) * 2)
    num_states = 2**nbit
    step = np.pi / (2**(nbit - 1))
    lookup_exp = np.exp(-1j * (np.arange(num_states) * step))
    
    nGenes = M * N
    Parent_B = np.random.randint(0, num_states, (nPop, M, N), dtype=np.int32)
    Parent_E = np.zeros(nPop, dtype=np.complex128)
    Parent_y = np.zeros(nPop, dtype=np.float64)
    
    for h in range(nPop):
        Parent_E[h] = np.sum(lookup_exp[Parent_B[h]] * W)
        Parent_y[h] = -np.abs(Parent_E[h])
        
    Offspring_B = np.zeros((nC, M, N), dtype=np.int32)
    Offspring_E = np.zeros(nC, dtype=np.complex128)
    Offspring_y = np.zeros(nC, dtype=np.float64)
    
    history = np.zeros(maxIt, dtype=np.float64)
    
    for It in range(maxIt):
        for j in range(nC // 2):
            idx1, idx2 = np.random.choice(nPop, 2, replace=False)
            p1_idx = idx1 if Parent_y[idx1] <= Parent_y[idx2] else idx2
            idx1, idx2 = np.random.choice(nPop, 2, replace=False)
            p2_idx = idx1 if Parent_y[idx1] <= Parent_y[idx2] else idx2
            
            p1 = Parent_B[p1_idx].ravel()
            p2 = Parent_B[p2_idx].ravel()
            
            pts = np.sort(np.random.choice(nGenes - 1, 2, replace=False))
            pt1, pt2 = pts[0] + 1, pts[1] + 1
            
            c1 = np.empty(nGenes, dtype=np.int32)
            c2 = np.empty(nGenes, dtype=np.int32)
            c1[:pt1] = p1[:pt1]; c1[pt1:pt2] = p2[pt1:pt2]; c1[pt2:] = p1[pt2:]
            c2[:pt1] = p2[:pt1]; c2[pt1:pt2] = p1[pt1:pt2]; c2[pt2:] = p2[pt2:]
            
            Offspring_B[2 * j] = c1.reshape(M, N)
            Offspring_B[2 * j + 1] = c2.reshape(M, N)
            
        for k in range(nC):
            Offspring_E[k] = np.sum(lookup_exp[Offspring_B[k]] * W)
            
        for k in range(nC):
            if np.random.rand() <= nMu:
                s_m = np.random.randint(0, M)
                s_n = np.random.randint(0, N)
                old_val = Offspring_B[k, s_m, s_n]
                new_val = np.random.randint(0, num_states)
                if new_val != old_val:
                    delta = (lookup_exp[new_val] - lookup_exp[old_val]) * W[s_m, s_n]
                    Offspring_E[k] += delta
                    Offspring_B[k, s_m, s_n] = new_val
            Offspring_y[k] = -np.abs(Offspring_E[k])
            
        all_B = np.concatenate([Parent_B, Offspring_B], axis=0)
        all_E = np.concatenate([Parent_E, Offspring_E], axis=0)
        all_y = np.concatenate([Parent_y, Offspring_y], axis=0)
        
        best_indices = np.argsort(all_y)[:nPop]
        Parent_B = all_B[best_indices]
        Parent_E = all_E[best_indices]
        Parent_y = all_y[best_indices]
        history[It] = Parent_y[0]
        
    elapsed = time.perf_counter() - t0
    return {
        'B': Parent_B[0],
        'fitness': Parent_y[0],
        'elapsed_sec': elapsed,
        'method': 'ga_exact',
        'evals': nPop + maxIt * nC,
        'history': history
    }

def design_metasurface(frequency_thz=1.7, period_um=70.0, theta=None,
                       phi=None, N=36, M=None, nbit=2,
                       method='fast', weights=None, output_path=None,
                       beam_type='plane', oam_charge=1, analyze_oam=True, **kwargs):
    """
    通用高层设计入口函数。
    
    参数:
      frequency_thz : 工作频率 (THz)，支持 0.1 - 5.0 THz
      period_um     : 单元周期 (μm)
      theta         : 目标俯仰角列表 (度)
      phi           : 目标方位角列表 (度)
      N             : 阵列列数 (当 M 省略时阵列为 N x N)
      M             : 阵列行数 (可选，默认等于 N)
      nbit          : 编码位数 (1 或 2)
      method        : 优化求解方法:
                      'fast' (默认) - 解析初始化 + 增量坐标下降
                      'analytical'  - 相位叠加与离散量化
                      'greedy'      - 纯坐标下降局部搜索
                      'ga_exact'    - 遗传算法搜索
      weights       : 目标方向复场幅度权重列表 (默认全 1)
      output_path   : 保存结果路径 (支持 .txt, .csv, .mat)
      beam_type     : plane 普通波束，vortex 为单束正向2-bit涡旋原型
      oam_charge    : 涡旋拓扑荷，目前支持 -2/-1/1/2
    """
    total_start = time.perf_counter()
    if beam_type not in ('plane', 'vortex'):
        raise ValueError("beam_type 必须为 plane 或 vortex")
    if theta is None:
        theta = [0.0] if beam_type == 'vortex' else [35.0, 45.0, 55.0]
    if phi is None:
        phi = [0.0] if beam_type == 'vortex' else [45.0, 135.0, 90.0]
    if M is None:
        M = N

    for name, value in (('N', N), ('M', M)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
            raise ValueError(f"{name} 必须为正整数")
    if isinstance(nbit, (bool, np.bool_)) or not isinstance(nbit, (int, np.integer)) or nbit not in (1, 2):
        raise ValueError("nbit 必须为 1 或 2")
    for name, value in (('frequency_thz', frequency_thz), ('period_um', period_um)):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须为有限正数")

    theta = list(np.atleast_1d(theta))
    phi = list(np.atleast_1d(phi))
    theta_values = np.asarray(theta, dtype=float)
    phi_values = np.asarray(phi, dtype=float)
    if theta_values.ndim != 1 or phi_values.ndim != 1 or len(theta) == 0 or len(theta) != len(phi):
        raise ValueError("theta 与 phi 必须为长度相同的非空一维列表")
    if not np.isfinite(theta_values).all() or not np.isfinite(phi_values).all():
        raise ValueError("波束角度必须为有限数值")
    if np.any((theta_values < 0) | (theta_values > 90)):
        raise ValueError("theta 必须在 0 到 90 度之间")
    theta = theta_values.tolist()
    phi = phi_values.tolist()
    if weights is not None:
        weight_values = np.asarray(weights, dtype=float)
        if (weight_values.ndim != 1 or len(weight_values) != len(theta)
                or not np.isfinite(weight_values).all() or np.any(weight_values < 0)
                or not np.any(weight_values > 0)):
            raise ValueError("weights 必须为匹配波束数量的有限非负幅度权重，且至少一项大于 0")
        weights = weight_values.tolist()
    if 'max_passes' in kwargs:
        passes = kwargs['max_passes']
        if isinstance(passes, bool) or not isinstance(passes, (int, np.integer)) or passes < 0:
            raise ValueError("max_passes 必须为非负整数")
    if method == 'ga_exact':
        for name, minimum in (('maxIt', 1), ('nPop', 2)):
            value = kwargs.get(name, 4000 if name == 'maxIt' else 50)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < minimum:
                raise ValueError(f"{name} 必须为不小于 {minimum} 的整数")
        for name, default in (('nPc', 0.9), ('nMu', 0.8)):
            value = kwargs.get(name, default)
            if not np.isscalar(value) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} 必须在 0 到 1 之间")
        if M * N < 3:
            raise ValueError("遗传算法的双点交叉至少需要 3 个阵元")
    
    if beam_type == 'vortex':
        if len(theta) != 1 or theta[0] != 0:
            raise ValueError("涡旋原型目前仅支持单束正向传播，请使用 theta=[0]、phi=[0]")
        from oam import design_vortex, analyze_vortex
        res = design_vortex(N=N, M=M, charge=oam_charge, nbit=nbit,
                            frequency_thz=frequency_thz, period_um=period_um,
                            method=method, **kwargs)
        if analyze_oam:
            analysis_start = time.perf_counter()
            res['oam_analysis'] = analyze_vortex(res['B'], oam_charge,
                frequency_thz=frequency_thz, period_um=period_um,
                uv_radius=res['observation_uv_radius'])
            res['analysis_elapsed_sec'] = time.perf_counter() - analysis_start
            res['oam_analysis']['metrics']['optimization'] = {
                'method': res['method'], 'initial_objective': res['initial_objective'],
                'final_objective': res['objective'], 'accepted_changes': res['accepted_changes'],
                'radial_samples': res['optimization_radial_samples'],
                'angular_samples': res['optimization_angular_samples'],
                'leakage_weight': res['leakage_weight'], 'null_weight': res['null_weight'],
                'seed': kwargs.get('seed', 42)}
    elif method == 'fast':
        res = design_hybrid(N, theta, phi, nbit=nbit, fre_thz=frequency_thz,
                            period_um=period_um, weights=weights, M=M, **kwargs)
    elif method == 'analytical':
        res = design_analytical(N, theta, phi, nbit=nbit, fre_thz=frequency_thz,
                                period_um=period_um, weights=weights, M=M)
    elif method == 'greedy':
        res = design_greedy(N, theta, phi, nbit=nbit, fre_thz=frequency_thz,
                            period_um=period_um, weights=weights, M=M, **kwargs)
    elif method == 'ga_exact':
        res = design_ga_exact(N, theta, phi, nbit=nbit, fre_thz=frequency_thz,
                              period_um=period_um, weights=weights, M=M, **kwargs)
    else:
        raise ValueError(f"未知方法: {method}，可选: 'fast', 'analytical', 'greedy', 'ga_exact'")
        
    B = res['B']
    
    if output_path:
        output_path = os.fspath(output_path)
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if output_path.endswith('.mat'):
            import scipy.io as sio # if available
            sio.savemat(output_path, {'B': B, 'fitness': res['fitness'], 'params': {
                'fre_thz': frequency_thz, 'period_um': period_um, 'theta': theta, 'phi': phi, 'nbit': nbit,
                'beam_type': beam_type, 'oam_charge': oam_charge if beam_type == 'vortex' else 0
            }})
        elif output_path.endswith('.csv'):
            np.savetxt(output_path, B, fmt='%d', delimiter=',')
        else: # .txt
            np.savetxt(output_path, B, fmt='%d', delimiter=',')
        res['output_path'] = output_path
        if beam_type == 'vortex' and 'oam_analysis' in res:
            from oam import save_vortex_analysis
            res['oam_files'] = save_vortex_analysis(res['oam_analysis'], output_path, B=B,
                                                  initial_B=res['initial_B'])

    if beam_type == 'vortex':
        res['total_elapsed_sec'] = time.perf_counter() - total_start
        
    return res

def calculate_farfield(B, theta_grid=None, phi_grid=None, fre_thz=1.7, period_um=70.0, nbit=2):
    """
    根据理想阵列因子计算指定角度网格上的二维远场。
    """
    if theta_grid is None:
        theta_grid = np.linspace(0, 90, 181)
    if phi_grid is None:
        phi_grid = np.linspace(0, 360, 361)
        
    fre_hz = float(fre_thz) * 1e12
    D_m = float(period_um) * 1e-6
    c = 1.0 / np.sqrt(EPSILON_0 * MU_0)
    w = fre_hz * 2.0 * np.pi
    k0 = w / c
    
    th_rad = np.deg2rad(theta_grid)
    ph_rad = np.deg2rad(phi_grid)
    
    sin_th = np.sin(th_rad)[:, None]
    cos_ph = np.cos(ph_rad)[None, :]
    sin_ph = np.sin(ph_rad)[None, :]
    u = (sin_th * cos_ph).ravel()
    v = (sin_th * sin_ph).ravel()
    
    M, N = B.shape
    m_idx = np.arange(1, M + 1) - 0.5
    n_idx = np.arange(1, N + 1) - 0.5
    X, Y = np.meshgrid(m_idx, n_idx, indexing='ij')
    
    fa = B * (np.pi / (2**(nbit - 1)))
    exp_fa_flat = np.exp(-1j * fa).ravel()
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    
    chunk_size = 5000
    total_pts = len(u)
    S_flat = np.zeros(total_pts, dtype=np.complex128)
    
    for i in range(0, total_pts, chunk_size):
        u_c = u[i:i+chunk_size]
        v_c = v[i:i+chunk_size]
        phase_c = k0 * D_m * (X_flat[:, None] * u_c[None, :] + Y_flat[:, None] * v_c[None, :])
        S_flat[i:i+chunk_size] = exp_fa_flat @ np.exp(-1j * phase_c)
        
    S = S_flat.reshape(len(theta_grid), len(phi_grid))
    abs_S = np.abs(S)
    return {
        'S': S,
        'abs_S': abs_S,
        'theta_grid': theta_grid,
        'phi_grid': phi_grid,
        'Smax': np.max(abs_S)
    }

def main():
    parser = argparse.ArgumentParser(description="超表面阵列编码设计与优化工具 (Metasurface Designer)")
    parser.add_argument("--freq", type=float, default=1.7, help="工作频率 (THz)，默认 1.7")
    parser.add_argument("--period", type=float, default=70.0, help="单元周期 (μm)，默认 70.0")
    parser.add_argument("--theta", type=float, nargs="+", default=None, help="目标俯仰角列表 (度)")
    parser.add_argument("--phi", type=float, nargs="+", default=None, help="目标方位角列表 (度)")
    parser.add_argument("--N", type=int, default=36, help="阵列边长 N (默认 36)")
    parser.add_argument("--M", type=int, default=None, help="阵列行数 M (默认等于 N)")
    parser.add_argument("--bit", type=int, default=2, choices=[1, 2], help="编码位数 (1 或 2，默认 2)")
    parser.add_argument("--method", type=str, default="fast", choices=["fast", "analytical", "greedy", "ga_exact"],
                        help="求解方法: fast(推荐), analytical, greedy, ga_exact")
    parser.add_argument("--weights", type=float, nargs="+", default=None, help="各波束权重列表")
    parser.add_argument("--output", type=str, default=None, help="输出文件路径 (.txt 或 .csv)")
    parser.add_argument("--beam-type", choices=['plane', 'vortex'], default='plane', help="普通波束或涡旋波")
    parser.add_argument("--oam-charge", type=int, choices=[-2, -1, 1, 2], default=1, help="涡旋拓扑荷")
    
    args = parser.parse_args()
    if args.theta is None:
        args.theta = [0.0] if args.beam_type == 'vortex' else [35.0, 45.0, 55.0]
    if args.phi is None:
        args.phi = [0.0] if args.beam_type == 'vortex' else [45.0, 135.0, 90.0]
    
    print("=" * 65)
    print("              超表面阵列编码优化设计系统 (Metasurface Designer)      ")
    print("=" * 65)
    print(f"工作频率: {args.freq} THz | 单元周期: {args.period} μm | 阵列尺寸: {args.M or args.N}x{args.N} | 编码位数: {args.bit}-bit")
    print(f"目标波束 theta: {args.theta}° | 方位角 phi: {args.phi}°")
    print(f"求解策略: {args.method}")
    print("-" * 65)
    
    res = design_metasurface(
        frequency_thz=args.freq,
        period_um=args.period,
        theta=args.theta,
        phi=args.phi,
        N=args.N,
        M=args.M,
        nbit=args.bit,
        method=args.method,
        weights=args.weights,
        output_path=args.output,
        beam_type=args.beam_type,
        oam_charge=args.oam_charge
    )
    
    print(f"求解耗时: {res['elapsed_sec']*1000.0:.2f} ms ({res['elapsed_sec']:.4f} s)")
    if args.beam_type == 'vortex':
        metrics = res['oam_analysis']['metrics']
        print(f"OAM 模式目标函数: {-res['fitness']:.6g} | 拓扑荷: {args.oam_charge:+d}")
        print(f"观测圆盘内目标模式占比: {metrics['target_mode_fraction']:.2%}")
        wide = metrics['wide_angle']
        print(f"较大圆盘 R={wide['uv_radius']:.2f} 内目标模式占比: {wide['target_mode_fraction']:.2%}（非总效率）")
        print(f"环上相位绕转: {metrics['ring_winding']} | 轴心/采样峰值: {metrics['axis_to_peak_db']:.2f} dB")
        print("环上绕转仅表示环内总荷，未验证奇点居中或未分裂；当前优化为坐标下降，无深度学习。")
        print(f"包含评价与导出的总耗时: {res['total_elapsed_sec']:.3f} s")
    else:
        print(f"适应度值: {res['fitness']:.2f} (有效相干合成场强: {-res['fitness']:.2f})")
    print(f"评价次数: {res['evals']}")
    print(f"编码状态分布: {np.unique(res['B'], return_counts=True)}")
    
    if args.output:
        print(f"编码矩阵已保存至: {args.output}")
    print("=" * 65)

if __name__ == '__main__':
    main()
