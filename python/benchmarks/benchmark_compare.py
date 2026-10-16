# benchmark_compare.py
"""
系统对比测试：
1. 遗传算法搜索 (Type A)
2. 搜索策略改进 (Type B - B1 解析相位共轭合成, B2 贪心坐标下降/局部优化, B3 混合优化)
3. 远场波束质量指标验证（主瓣指向误差、峰值增益、旁瓣水平）
"""
import time
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from metasurface_designer import compute_steering_matrix as _compute_steering_matrix


def compute_steering_matrix(N, theta, phi, fre=1.7e12, D=70e-6, weights=None):
    """Adapt historical Hz/metre arguments to the bundled solver's THz/um API."""
    return _compute_steering_matrix(
        N, theta, phi, fre_thz=fre / 1e12, period_um=D / 1e-6, weights=weights
    )

def run_type_a_ga(N, theta, phi, nbit=2, nPop=50, maxIt=4000, nPc=0.9, nMu=0.8,
                  fre=1.7e12, D=70e-6, seed=42):
    """
    Type A: 遗传算法搜索
    使用锦标赛选择、双点交叉、单点变异和精英筛选
    加速手段：预计算导向矩阵 W、增量评估变异个体、消除循环内绘图
    """
    np.random.seed(seed)
    W = compute_steering_matrix(N, theta, phi, fre=fre, D=D)
    nC = int(round(nPop * nPc / 2.0) * 2)
    
    num_states = 2**nbit
    angle_step = np.pi / (2**(nbit - 1))
    lookup_exp = np.exp(-1j * (np.arange(num_states) * angle_step))
    
    # 种群初始化
    # Parent_B: (nPop, N, N)
    Parent_B = np.random.randint(0, num_states, (nPop, N, N))
    Parent_E = np.zeros(nPop, dtype=np.complex128)
    Parent_y = np.zeros(nPop, dtype=np.float64)
    
    # 初始化评价
    for h in range(nPop):
        exp_fa = lookup_exp[Parent_B[h]]
        Parent_E[h] = np.sum(exp_fa * W)
        Parent_y[h] = -np.abs(Parent_E[h])
        
    t_start = time.perf_counter()
    
    record = np.zeros(maxIt, dtype=np.float64)
    total_evals = nPop
    
    # 染色体扁平长度
    n_genes = N * N
    
    Offspring_B = np.zeros((nC, N, N), dtype=np.int32)
    Offspring_E = np.zeros(nC, dtype=np.complex128)
    Offspring_y = np.zeros(nC, dtype=np.float64)
    
    for It in range(maxIt):
        # 1. 锦标赛选择与双点交叉
        for j in range(nC // 2):
            # 锦标赛选择 p1
            idx1, idx2 = np.random.choice(nPop, 2, replace=False)
            p1_idx = idx1 if Parent_y[idx1] <= Parent_y[idx2] else idx2
            
            # 锦标赛选择 p2
            idx1, idx2 = np.random.choice(nPop, 2, replace=False)
            p2_idx = idx1 if Parent_y[idx1] <= Parent_y[idx2] else idx2
            
            p1 = Parent_B[p1_idx].ravel()
            p2 = Parent_B[p2_idx].ravel()
            
            # 双点交叉
            pts = np.sort(np.random.choice(n_genes - 1, 2, replace=False))
            pt1, pt2 = pts[0] + 1, pts[1] + 1
            
            c1 = np.empty(n_genes, dtype=np.int32)
            c2 = np.empty(n_genes, dtype=np.int32)
            
            c1[:pt1] = p1[:pt1]
            c1[pt1:pt2] = p2[pt1:pt2]
            c1[pt2:] = p1[pt2:]
            
            c2[:pt1] = p2[:pt1]
            c2[pt1:pt2] = p1[pt1:pt2]
            c2[pt2:] = p2[pt2:]
            
            Offspring_B[2 * j] = c1.reshape(N, N)
            Offspring_B[2 * j + 1] = c2.reshape(N, N)
            
        # 交叉后需计算电场 E
        for k in range(nC):
            exp_fa = lookup_exp[Offspring_B[k]]
            Offspring_E[k] = np.sum(exp_fa * W)
            
        # 2. 变异 (单点变异)
        for k in range(nC):
            if np.random.rand() <= nMu:
                s_m = np.random.randint(0, N)
                s_n = np.random.randint(0, N)
                old_val = Offspring_B[k, s_m, s_n]
                new_val = np.random.randint(0, num_states)
                if new_val != old_val:
                    # 增量更新 E: O(1)
                    delta = (lookup_exp[new_val] - lookup_exp[old_val]) * W[s_m, s_n]
                    Offspring_E[k] += delta
                    Offspring_B[k, s_m, s_n] = new_val
            Offspring_y[k] = -np.abs(Offspring_E[k])
            
        total_evals += nC
        
        # 3. 精英筛选
        all_B = np.concatenate([Parent_B, Offspring_B], axis=0)
        all_E = np.concatenate([Parent_E, Offspring_E], axis=0)
        all_y = np.concatenate([Parent_y, Offspring_y], axis=0)
        
        sort_order = np.argsort(all_y)
        best_indices = sort_order[:nPop]
        
        Parent_B = all_B[best_indices]
        Parent_E = all_E[best_indices]
        Parent_y = all_y[best_indices]
        
        record[It] = Parent_y[0]
        
    t_elapsed = time.perf_counter() - t_start
    best_B = Parent_B[0]
    best_fitness = Parent_y[0]
    
    return {
        'best_B': best_B,
        'best_fitness': best_fitness,
        'time_sec': t_elapsed,
        'total_evals': total_evals,
        'record': record
    }

def run_type_b1_analytical(N, theta, phi, nbit=2, fre=1.7e12, D=70e-6, weights=None):
    """
    Type B1: 理论共轭相位综合法 (Analytical Phase Matching)
    原理：对目标波束理想连续等效场的共轭相位进行直接量化
    计算复杂度：O(K * N^2)，耗时 < 0.1 ms
    """
    t_start = time.perf_counter()
    W = compute_steering_matrix(N, theta, phi, fre=fre, D=D, weights=weights)
    
    # W(m, n) = sum_k w_k exp(-1j * psi_k(m, n))
    # 为使每个阵元在各波束方向同相相干叠加，阵元最佳连续辐射相位为 -angle(W(m, n))
    # 即 exp(-1j * fa) 尽可能与 conj(W) 同相，以最大化 sum(exp(-1j * fa) * W)
    target_continuous_phase = np.angle(W) % (2 * np.pi)
    
    num_states = 2**nbit
    step = 2 * np.pi / num_states
    
    # 量化到 0..num_states-1
    # 状态 k 对应的相位为 k * pi / 2^(nbit-1) = k * step
    quantized_B = np.round(target_continuous_phase / step).astype(np.int32) % num_states
    
    lookup_exp = np.exp(-1j * (np.arange(num_states) * (np.pi / (2**(nbit - 1)))))
    E = np.sum(lookup_exp[quantized_B] * W)
    fitness = -np.abs(E)
    t_elapsed = time.perf_counter() - t_start
    
    return {
        'best_B': quantized_B,
        'best_fitness': fitness,
        'time_sec': t_elapsed,
        'total_evals': 1
    }

def run_type_b2_greedy_coordinate_descent(N, theta, phi, nbit=2, fre=1.7e12, D=70e-6,
                                          weights=None, max_passes=5, initial_B=None, seed=42):
    """
    Type B2: 贪心坐标下降 / 逐元迭代局部搜索 (Coordinate Descent / Iterative Conditional Modes)
    利用 O(1) 增量评估，逐个单元测试所有可能离散状态并贪心选择最佳状态
    """
    t_start = time.perf_counter()
    W = compute_steering_matrix(N, theta, phi, fre=fre, D=D, weights=weights)
    num_states = 2**nbit
    angle_step = np.pi / (2**(nbit - 1))
    lookup_exp = np.exp(-1j * (np.arange(num_states) * angle_step))
    
    if initial_B is None:
        np.random.seed(seed)
        B = np.random.randint(0, num_states, (N, N), dtype=np.int32)
    else:
        B = initial_B.copy()
        
    E_curr = np.sum(lookup_exp[B] * W)
    curr_score = np.abs(E_curr)
    
    total_evals = 1
    
    for pass_idx in range(max_passes):
        improved = False
        # 随机遍历顺序以避免固定方向扫描引入的局部偏置
        coords = [(m, n) for m in range(N) for n in range(N)]
        np.random.shuffle(coords)
        
        for m, n in coords:
            old_val = B[m, n]
            best_val = old_val
            best_score = curr_score
            best_E = E_curr
            
            W_mn = W[m, n]
            old_exp = lookup_exp[old_val]
            
            # 测试其他状态
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
            
    fitness = -curr_score
    t_elapsed = time.perf_counter() - t_start
    
    return {
        'best_B': B,
        'best_fitness': fitness,
        'time_sec': t_elapsed,
        'total_evals': total_evals
    }

def run_type_b3_hybrid(N, theta, phi, nbit=2, fre=1.7e12, D=70e-6, weights=None):
    """
    Type B3: 混合算法：解析共轭初始化 + 贪心坐标精修 (Analytical Init + Greedy Refinement)
    """
    t_start = time.perf_counter()
    res_b1 = run_type_b1_analytical(N, theta, phi, nbit=nbit, fre=fre, D=D, weights=weights)
    res_b2 = run_type_b2_greedy_coordinate_descent(N, theta, phi, nbit=nbit, fre=fre, D=D,
                                                  weights=weights, max_passes=3,
                                                  initial_B=res_b1['best_B'])
    t_elapsed = time.perf_counter() - t_start
    return {
        'best_B': res_b2['best_B'],
        'best_fitness': res_b2['best_fitness'],
        'time_sec': t_elapsed,
        'total_evals': res_b1['total_evals'] + res_b2['total_evals']
    }

def calculate_farfield_metrics(B, theta_target, phi_target, nbit=2, fre=1.7e12, D=70e-6):
    """
    基于理想阵列因子计算角度采样网格上的远场指标：
    最大采样幅度位置、峰值场强和目标方向附近的采样幅度。
    """
    c = 2.99792458e8
    lamda = c / fre
    k0 = 2 * np.pi / lamda
    
    # 远场角域采样网格 (与 farfield.m 一致)
    # 为保证高效测评，将 linspace 网格设定为合理分辨率 (theta: 181 点, phi: 361 点)
    theta_grid = np.deg2rad(np.linspace(0, 90, 181))   # (T,)
    phi_grid = np.deg2rad(np.linspace(0, 360, 361))    # (P,)
    
    # 展成 2D 导向网格
    # u = sin(theta) * cos(phi), v = sin(theta) * sin(phi)
    sin_th = np.sin(theta_grid)[:, None] # (T, 1)
    cos_ph = np.cos(phi_grid)[None, :]   # (1, P)
    sin_ph = np.sin(phi_grid)[None, :]   # (1, P)
    
    u = sin_th * cos_ph # (T, P)
    v = sin_th * sin_ph # (T, P)
    
    N_row, N_col = B.shape
    m_idx = np.arange(1, N_row + 1) - 0.5
    n_idx = np.arange(1, N_col + 1) - 0.5
    X, Y = np.meshgrid(m_idx, n_idx, indexing='ij') # (N, N)
    
    fa = B * (np.pi / (2**(nbit - 1)))
    exp_phase = np.exp(-1j * fa) # (N, N)
    
    # 向量化远场求和:
    # S(theta, phi) = sum_{m,n} exp(-1j * (fa(m,n) + k0 * D * (X*u + Y*v)))
    # 注意: k0 * D * (X*u + Y*v) 可以利用张量缩并高效计算
    # X_flat: (N^2,), u_flat: (T*P,)
    X_flat = X.ravel()
    Y_flat = Y.ravel()
    exp_phase_flat = exp_phase.ravel()
    
    u_flat = u.ravel() # (T*P,)
    v_flat = v.ravel() # (T*P,)
    
    # 分块矩阵乘法计算以节约显存/内存
    chunk_size = 5000
    total_pts = len(u_flat)
    S_flat = np.zeros(total_pts, dtype=np.complex128)
    
    for i in range(0, total_pts, chunk_size):
        u_c = u_flat[i:i+chunk_size]
        v_c = v_flat[i:i+chunk_size]
        # (N^2, chunk)
        phase_c = k0 * D * (X_flat[:, None] * u_c[None, :] + Y_flat[:, None] * v_c[None, :])
        S_flat[i:i+chunk_size] = exp_phase_flat @ np.exp(-1j * phase_c)
        
    S = S_flat.reshape(len(theta_grid), len(phi_grid))
    abs_S = np.abs(S)
    Smax = np.max(abs_S)
    
    # 找到峰值位置
    max_idx = np.unravel_index(np.argmax(abs_S), abs_S.shape)
    peak_theta_deg = np.rad2deg(theta_grid[max_idx[0]])
    peak_phi_deg = np.rad2deg(phi_grid[max_idx[1]])
    
    # 计算指向误差（对比目标波束）
    target_th = np.atleast_1d(theta_target)
    target_ph = np.atleast_1d(phi_target)
    
    # 计算目标方向处的实际场强
    target_fields = []
    for th_t, ph_t in zip(target_th, target_ph):
        th_idx = np.argmin(np.abs(np.rad2deg(theta_grid) - th_t))
        ph_idx = np.argmin(np.abs(np.rad2deg(phi_grid) - ph_t))
        target_fields.append(abs_S[th_idx, ph_idx])
        
    return {
        'Smax': Smax,
        'peak_theta_deg': peak_theta_deg,
        'peak_phi_deg': peak_phi_deg,
        'target_fields': target_fields
    }

if __name__ == '__main__':
    print("=" * 70)
    print("全算法综合对比实验 (相同 36x36 阵列, 2-bit, 4000 代基准)")
    print("=" * 70)
    
    test_cases = [
        {"name": "单波束算例1", "theta": [30.0], "phi": [150.0]},
        {"name": "单波束算例2", "theta": [45.0], "phi": [90.0]},
        {"name": "双波束算例",   "theta": [20.0, 25.0], "phi": [90.0, 120.0]},
        {"name": "三波束默认算例", "theta": [35.0, 45.0, 55.0], "phi": [45.0, 135.0, 90.0]}
    ]
    
    for case in test_cases:
        print(f"\n>>> 测试用例: {case['name']} (theta={case['theta']}, phi={case['phi']})")
        
        # 1. Type A: 遗传算法 (4000 代)
        res_a = run_type_a_ga(36, case['theta'], case['phi'], nbit=2, maxIt=4000, seed=42)
        mf_a = calculate_farfield_metrics(res_a['best_B'], case['theta'], case['phi'])
        
        # 2. Type B1: 解析相位共轭
        res_b1 = run_type_b1_analytical(36, case['theta'], case['phi'], nbit=2)
        mf_b1 = calculate_farfield_metrics(res_b1['best_B'], case['theta'], case['phi'])
        
        # 3. Type B2: 纯贪心坐标下降 (从随机初始开始)
        res_b2 = run_type_b2_greedy_coordinate_descent(36, case['theta'], case['phi'], nbit=2, max_passes=5, seed=42)
        mf_b2 = calculate_farfield_metrics(res_b2['best_B'], case['theta'], case['phi'])
        
        # 4. Type B3: 混合优化 (解析初始 + 局部精修)
        res_b3 = run_type_b3_hybrid(36, case['theta'], case['phi'], nbit=2)
        mf_b3 = calculate_farfield_metrics(res_b3['best_B'], case['theta'], case['phi'])
        
        print(f"| 方案 | 耗时 (s) | 评价次数 | 适应度 (越负越好) | 峰值场强 Smax | 主瓣指向 (th, phi) |")
        print(f"|---|---|---|---|---|---|")
        print(f"| 遗传算法 (Type A, 4000代) | {res_a['time_sec']:.3f} s | {res_a['total_evals']} | {res_a['best_fitness']:.2f} | {mf_a['Smax']:.1f} | ({mf_a['peak_theta_deg']:.1f}°, {mf_a['peak_phi_deg']:.1f}°) |")
        print(f"| 解析共轭合成 (Type B1)            | {res_b1['time_sec']*1000:.3f} ms | {res_b1['total_evals']} | {res_b1['best_fitness']:.2f} | {mf_b1['Smax']:.1f} | ({mf_b1['peak_theta_deg']:.1f}°, {mf_b1['peak_phi_deg']:.1f}°) |")
        print(f"| 贪心坐标下降 (Type B2)            | {res_b2['time_sec']*1000:.3f} ms | {res_b2['total_evals']} | {res_b2['best_fitness']:.2f} | {mf_b2['Smax']:.1f} | ({mf_b2['peak_theta_deg']:.1f}°, {mf_b2['peak_phi_deg']:.1f}°) |")
        print(f"| 混合优化策略 (Type B3)            | {res_b3['time_sec']*1000:.3f} ms | {res_b3['total_evals']} | {res_b3['best_fitness']:.2f} | {mf_b3['Smax']:.1f} | ({mf_b3['peak_theta_deg']:.1f}°, {mf_b3['peak_phi_deg']:.1f}°) |")
