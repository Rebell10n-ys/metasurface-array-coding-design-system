# verify_pipeline.py
import os
import sys
from pathlib import Path
import numpy as np

# 导入设计工具
from metasurface_designer import design_metasurface

# 1. 运行生成
out_txt = Path(__file__).resolve().parents[1] / 'output' / 'test_verification.txt'
res = design_metasurface(
    frequency_thz=1.7,
    period_um=70.0,
    theta=[30.0],
    phi=[150.0],
    N=36,
    nbit=2,
    method='fast',
    output_path=out_txt
)

print(f"1. Python 生成完成！")
print(f"   求解耗时: {res['elapsed_sec'] * 1000.0:.2f} ms")
print(f"   适应度值: {res['fitness']:.2f}")

# 2. 验证 TXT 文件内容
if not os.path.exists(out_txt):
    print("错误: 文件未生成！")
    sys.exit(1)

with open(out_txt, 'r', encoding='utf-8') as f:
    lines = [line.strip().split(',') for line in f if line.strip()]

mat = np.array(lines, dtype=int)
print(f"2. TXT 文件读取与数据校验:")
print(f"   矩阵尺寸: {mat.shape} (预期 36x36)")
print(f"   取值集合: {np.unique(mat)} (预期 [0 1 2 3])")

assert mat.shape == (36, 36), "矩阵尺寸不正确！"
assert set(np.unique(mat)).issubset({0, 1, 2, 3}), "编码值不合法！"
print(f"   >> TXT 数据结构与编码格式 100% 合规！")
