# run_full_benchmarks.py
"""
全自动基准测试套件：
覆盖单波束/多波束、1-bit/2-bit、不同阵列尺寸 (30x30, 36x36, 48x48)、不同频率 (0.7, 1.7, 3.0 THz)
对比 Type A (遗传算法)、Type B1 (解析合成)、Type B2 (贪心下降)、Type B3 (混合优化)
输出完整对照数据与远场波束质量指标。
"""
import os
import sys
import time
import json
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / 'python'))
from metasurface_designer import (
    compute_steering_matrix,
    evaluate_fitness,
    design_metasurface,
    calculate_farfield,
    design_analytical,
    design_greedy,
    design_hybrid,
    design_ga_exact
)

def run_suite():
    report_dir = PROJECT_ROOT / 'reports'
    report_dir.mkdir(exist_ok=True)
    
    scenarios = [
        {
            "id": "single_36x36_2bit",
            "desc": "单波束基准 (36x36, 2-bit, 1.7 THz, theta=30, phi=150)",
            "N": 36, "nbit": 2, "freq": 1.7, "period": 70.0,
            "theta": [30.0], "phi": [150.0]
        },
        {
            "id": "single_36x36_1bit",
            "desc": "单波束 1-bit 基准 (36x36, 1-bit, 1.7 THz, theta=45, phi=90)",
            "N": 36, "nbit": 1, "freq": 1.7, "period": 70.0,
            "theta": [45.0], "phi": [90.0]
        },
        {
            "id": "dual_36x36_2bit",
            "desc": "双波束基准 (36x36, 2-bit, 1.7 THz, theta=[20,25], phi=[90,120])",
            "N": 36, "nbit": 2, "freq": 1.7, "period": 70.0,
            "theta": [20.0, 25.0], "phi": [90.0, 120.0]
        },
        {
            "id": "triple_36x36_2bit_default",
            "desc": "原版默认三波束 (36x36, 2-bit, 1.7 THz, theta=[35,45,55], phi=[45,135,90])",
            "N": 36, "nbit": 2, "freq": 1.7, "period": 70.0,
            "theta": [35.0, 45.0, 55.0], "phi": [45.0, 135.0, 90.0]
        },
        {
            "id": "scale_30x30_2bit",
            "desc": "小规模阵列 (30x30, 2-bit, 1.7 THz, theta=35, phi=45)",
            "N": 30, "nbit": 2, "freq": 1.7, "period": 70.0,
            "theta": [35.0], "phi": [45.0]
        },
        {
            "id": "scale_48x48_2bit",
            "desc": "大规模阵列 (48x48, 2-bit, 1.7 THz, theta=30, phi=60)",
            "N": 48, "nbit": 2, "freq": 1.7, "period": 70.0,
            "theta": [30.0], "phi": [60.0]
        },
        {
            "id": "freq_0.7THz_239um",
            "desc": "历史算例 duoboshu (36x36, 2-bit, 0.7 THz, period=239.6um, theta=30, phi=90)",
            "N": 36, "nbit": 2, "freq": 0.7, "period": 239.6,
            "theta": [30.0], "phi": [90.0]
        }
    ]
    
    results = []
    
    print("=" * 80)
    print("                  超表面设计优化全套综合基准评测                  ")
    print("=" * 80)
    
    for sc in scenarios:
        print(f"\n[测试项] {sc['desc']}")
        N = sc['N']
        nbit = sc['nbit']
        freq = sc['freq']
        period = sc['period']
        theta = sc['theta']
        phi = sc['phi']
        
        # 1. Type A (遗传算法, 4000 代)
        t0 = time.perf_counter()
        res_a = design_ga_exact(N, theta, phi, nbit=nbit, fre_thz=freq, period_um=period, maxIt=4000, seed=42)
        t_a = time.perf_counter() - t0
        ff_a = calculate_farfield(res_a['B'], fre_thz=freq, period_um=period, nbit=nbit)
        
        # 2. Type B1 (解析合成)
        t0 = time.perf_counter()
        res_b1 = design_analytical(N, theta, phi, nbit=nbit, fre_thz=freq, period_um=period)
        t_b1 = time.perf_counter() - t0
        ff_b1 = calculate_farfield(res_b1['B'], fre_thz=freq, period_um=period, nbit=nbit)
        
        # 3. Type B2 (贪心坐标下降)
        t0 = time.perf_counter()
        res_b2 = design_greedy(N, theta, phi, nbit=nbit, fre_thz=freq, period_um=period, max_passes=4, seed=42)
        t_b2 = time.perf_counter() - t0
        ff_b2 = calculate_farfield(res_b2['B'], fre_thz=freq, period_um=period, nbit=nbit)
        
        # 4. Type B3 (推荐混合优化)
        t0 = time.perf_counter()
        res_b3 = design_hybrid(N, theta, phi, nbit=nbit, fre_thz=freq, period_um=period)
        t_b3 = time.perf_counter() - t0
        ff_b3 = calculate_farfield(res_b3['B'], fre_thz=freq, period_um=period, nbit=nbit)
        
        row_data = {
            "scenario": sc,
            "TypeA_GA": {
                "time_sec": t_a,
                "fitness": float(res_a['fitness']),
                "evals": res_a['evals'],
                "Smax": float(ff_a['Smax'])
            },
            "TypeB1_Analytical": {
                "time_sec": t_b1,
                "fitness": float(res_b1['fitness']),
                "evals": res_b1['evals'],
                "Smax": float(ff_b1['Smax'])
            },
            "TypeB2_Greedy": {
                "time_sec": t_b2,
                "fitness": float(res_b2['fitness']),
                "evals": res_b2['evals'],
                "Smax": float(ff_b2['Smax'])
            },
            "TypeB3_Hybrid": {
                "time_sec": t_b3,
                "fitness": float(res_b3['fitness']),
                "evals": res_b3['evals'],
                "Smax": float(ff_b3['Smax'])
            }
        }
        results.append(row_data)
        
        print(f"  Type A (4000代GA): 耗时 {t_a:.3f} s, 适应度 {res_a['fitness']:.2f}, Smax {ff_a['Smax']:.1f}")
        print(f"  Type B1 (解析共轭): 耗时 {t_b1*1000:.3f} ms, 适应度 {res_b1['fitness']:.2f}, Smax {ff_b1['Smax']:.1f}")
        print(f"  Type B2 (贪心下降): 耗时 {t_b2*1000:.3f} ms, 适应度 {res_b2['fitness']:.2f}, Smax {ff_b2['Smax']:.1f}")
        print(f"  Type B3 (混合优化): 耗时 {t_b3*1000:.3f} ms, 适应度 {res_b3['fitness']:.2f}, Smax {ff_b3['Smax']:.1f}")
        
    # 保存 JSON 数据
    report_path = report_dir / 'benchmark_summary.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 测试数据已完整保存至 {report_path}")

if __name__ == '__main__':
    run_suite()
