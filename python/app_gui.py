#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=============================================================================
超表面阵列编码设计系统 - 图形用户界面 (Metasurface Designer GUI)
=============================================================================
基于 Python Tkinter 构建的原生桌面客户端，支持：
1. 参数配置：频率、周期、行列数、比特位数、波束角度列表、权重与算法选择
2. 毫秒级求解计算与进度反馈
3. 阵列编码状态热力图实时可视化
4. 导出逗号分隔的整数编码 TXT 文件
=============================================================================
"""

import os
import sys
import time
import datetime
import threading
from pathlib import Path
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk

APP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = APP_DIR.parent if APP_DIR.name == 'python' else APP_DIR

# GUI 与命令行共用同目录下的求解内核。
try:
    from metasurface_designer import design_metasurface
except ImportError:
    messagebox.showerror("错误", "未找到 python/metasurface_designer.py 模块或其依赖！")
    sys.exit(1)

class MetasurfaceApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("超表面阵列编码设计系统 (Metasurface Designer)")
        self.geometry("1080x860")
        self.minsize(980, 800)
        
        # 结果缓存
        self.current_B = None
        self.current_info = {}
        self._vortex_active = False
        
        # 界面风格配置
        self.style = ttk.Style(self)
        try:
            self.style.theme_use('vista')
        except Exception:
            pass
            
        self._init_ui()
        self._update_default_filename()

    def _init_ui(self):
        # 顶部标题栏
        top_frame = ttk.Frame(self, padding="10 10 10 5")
        top_frame.pack(fill=tk.X)
        
        title_lbl = ttk.Label(top_frame, text="超表面阵列离散编码设计系统", font=("Microsoft YaHei", 16, "bold"))
        title_lbl.pack(side=tk.LEFT)
        sub_lbl = ttk.Label(top_frame, text="面向太赫兹频段 (0.1–5 THz) 1/2-bit 阵列波束赋形", font=("Microsoft YaHei", 9), foreground="#666666")
        sub_lbl.pack(side=tk.LEFT, padx=15, pady=5)
        
        # 主体分为左右两侧面板
        main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        left_panel = ttk.Frame(main_paned, padding=5)
        right_panel = ttk.Frame(main_paned, padding=5)
        main_paned.add(left_panel, weight=4)
        main_paned.add(right_panel, weight=5)
        
        # 核心物理与阵列参数
        param_frame = ttk.LabelFrame(left_panel, text="设计需求与物理参数", padding=10)
        param_frame.pack(fill=tk.X, pady=4)
        
        # Grid 配置
        row = 0
        ttk.Label(param_frame, text="工作频率 (THz):").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.ent_freq = ttk.Entry(param_frame, width=12)
        self.ent_freq.insert(0, "1.7")
        self.ent_freq.grid(row=row, column=1, sticky=tk.EW, pady=3, padx=5)
        
        ttk.Label(param_frame, text="单元周期 (μm):").grid(row=row, column=2, sticky=tk.W, pady=3)
        self.ent_period = ttk.Entry(param_frame, width=12)
        self.ent_period.insert(0, "70.0")
        self.ent_period.grid(row=row, column=3, sticky=tk.EW, pady=3, padx=5)
        
        row += 1
        ttk.Label(param_frame, text="阵列尺寸 (NxN):").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.ent_N = ttk.Entry(param_frame, width=12)
        self.ent_N.insert(0, "36")
        self.ent_N.grid(row=row, column=1, sticky=tk.EW, pady=3, padx=5)
        
        ttk.Label(param_frame, text="编码比特位数:").grid(row=row, column=2, sticky=tk.W, pady=3)
        self.combo_bit = ttk.Combobox(param_frame, values=["2-bit (0,1,2,3)", "1-bit (0,1)"], state="readonly", width=11)
        self.combo_bit.grid(row=row, column=3, sticky=tk.EW, pady=3, padx=5)
        self.combo_bit.current(0)

        row += 1
        ttk.Label(param_frame, text="波束类型:").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.combo_beam_type = ttk.Combobox(param_frame, values=['普通波束', '涡旋波 (OAM)'], state='readonly', width=14)
        self.combo_beam_type.grid(row=row, column=1, sticky=tk.EW, padx=5, pady=3)
        self.combo_beam_type.current(0)
        self.combo_beam_type.bind('<<ComboboxSelected>>', self._on_beam_type_change)
        ttk.Label(param_frame, text="拓扑荷 l:").grid(row=row, column=2, sticky=tk.W, pady=3)
        self.combo_charge = ttk.Combobox(param_frame, values=['-2', '-1', '+1', '+2'], state='disabled', width=10)
        self.combo_charge.grid(row=row, column=3, sticky=tk.EW, padx=5, pady=3)
        self.combo_charge.set('+1')
        self.combo_charge.bind('<<ComboboxSelected>>', lambda event: self._update_default_filename())
        
        row += 1
        ttk.Label(param_frame, text="目标俯仰角 θ (°):").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.ent_theta = ttk.Entry(param_frame)
        self.ent_theta.insert(0, "30")
        self.ent_theta.grid(row=row, column=1, columnspan=3, sticky=tk.EW, pady=3, padx=5)
        
        row += 1
        ttk.Label(param_frame, text="目标方位角 φ (°):").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.ent_phi = ttk.Entry(param_frame)
        self.ent_phi.insert(0, "150")
        self.ent_phi.grid(row=row, column=1, columnspan=3, sticky=tk.EW, pady=3, padx=5)
        
        row += 1
        ttk.Label(param_frame, text="波束幅度权重 (可选):").grid(row=row, column=0, sticky=tk.W, pady=3)
        self.ent_weights = ttk.Entry(param_frame)
        self.ent_weights.grid(row=row, column=1, columnspan=3, sticky=tk.EW, pady=3, padx=5)
        
        ttk.Label(param_frame, text="普通波束用逗号或空格分隔角度；OAM原型：正向、2-bit、偶数方阵。", font=("Microsoft YaHei", 8), foreground="#888888", wraplength=450).grid(row=row+1, column=0, columnspan=4, sticky=tk.W, pady=2)
        
        # 算法配置
        algo_frame = ttk.LabelFrame(left_panel, text="算法求解设置", padding=10)
        algo_frame.pack(fill=tk.X, pady=4)
        
        ttk.Label(algo_frame, text="求解模式:").pack(anchor=tk.W)
        self.algo_var = tk.StringVar(value="fast")
        ttk.Radiobutton(algo_frame, text="混合优化 (fast) —— 解析初值 + 坐标下降微调",
                        variable=self.algo_var, value="fast").pack(anchor=tk.W, pady=2)
        ttk.Radiobutton(algo_frame, text="解析综合 (analytical) —— 相位叠加与离散量化",
                        variable=self.algo_var, value="analytical").pack(anchor=tk.W, pady=2)
        self.ga_radio = ttk.Radiobutton(algo_frame, text="遗传算法 (ga_exact) —— 种群迭代搜索",
                        variable=self.algo_var, value="ga_exact")
        self.ga_radio.pack(anchor=tk.W, pady=2)
        
        # 自主编辑 GA 迭代次数
        ga_sub_frame = ttk.Frame(algo_frame)
        ga_sub_frame.pack(fill=tk.X, padx=20, pady=2)
        ttk.Label(ga_sub_frame, text="└ GA 最大迭代代数:").pack(side=tk.LEFT)
        self.ent_maxIt = ttk.Entry(ga_sub_frame, width=8)
        self.ent_maxIt.insert(0, "4000")
        self.ent_maxIt.pack(side=tk.LEFT, padx=5)
        ttk.Label(ga_sub_frame, text="(仅限 ga_exact 模式生效，可自由修改)", font=("Microsoft YaHei", 8), foreground="#888888").pack(side=tk.LEFT)
        
        # 导出文件设置
        file_frame = ttk.LabelFrame(left_panel, text="输出 TXT 文件设置", padding=10)
        file_frame.pack(fill=tk.X, pady=4)
        
        self.file_path_var = tk.StringVar()
        file_sub = ttk.Frame(file_frame)
        file_sub.pack(fill=tk.X)
        self.ent_path = ttk.Entry(file_sub, textvariable=self.file_path_var)
        self.ent_path.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        ttk.Button(file_sub, text="浏览...", command=self._choose_save_path, width=8).pack(side=tk.RIGHT)
        
        # 执行设计大按钮
        btn_frame = ttk.Frame(left_panel, padding=5)
        btn_frame.pack(fill=tk.X, pady=6)
        
        self.btn_run = tk.Button(btn_frame, text="▶ 立即生成编码矩阵", font=("Microsoft YaHei", 11, "bold"),
                                 bg="#0078D7", fg="white", activebackground="#005A9E", activeforeground="white",
                                 relief=tk.FLAT, height=2, command=self._run_optimization)
        self.btn_run.pack(fill=tk.X)
        
        # ================= 右侧：矩阵可视化与指标展示 =================
        right_lbl_frame = ttk.LabelFrame(right_panel, text="编码矩阵实时预览与指标", padding=10)
        right_lbl_frame.pack(fill=tk.BOTH, expand=True)
        
        # 状态指示与计时
        self.stat_frame = ttk.Frame(right_lbl_frame)
        self.stat_frame.pack(fill=tk.X, pady=2)
        
        self.lbl_time = ttk.Label(self.stat_frame, text="耗时: -- ms", font=("Microsoft YaHei", 10, "bold"), foreground="#0078D7")
        self.lbl_time.pack(side=tk.LEFT, padx=5)
        
        self.lbl_fitness = ttk.Label(self.stat_frame, text="适应度: --", font=("Microsoft YaHei", 10, "bold"))
        self.lbl_fitness.pack(side=tk.LEFT, padx=15)
        
        self.lbl_evals = ttk.Label(self.stat_frame, text="评价次数: --", font=("Microsoft YaHei", 9), foreground="#666666")
        self.lbl_evals.pack(side=tk.RIGHT, padx=5)
        
        # 图像展示 Canvas
        self.canvas_frame = ttk.Frame(right_lbl_frame)
        self.canvas_frame.pack(fill=tk.BOTH, expand=True, pady=8)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="#F0F0F0", highlightthickness=1, highlightbackground="#CCCCCC")
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # 底部色条说明与阵列统计
        self.info_lbl = ttk.Label(right_lbl_frame, text="请配置参数后点击【立即生成编码矩阵】", font=("Microsoft YaHei", 9), wraplength=540)
        self.info_lbl.pack(fill=tk.X, pady=4)
        
        btn_action_frame = ttk.Frame(right_lbl_frame)
        btn_action_frame.pack(fill=tk.X, pady=4)
        
        self.btn_open_file = ttk.Button(btn_action_frame, text="打开生成的 TXT", command=self._open_result_file, state=tk.DISABLED)
        self.btn_open_file.pack(side=tk.LEFT, padx=5)
        
        self.btn_open_dir = ttk.Button(btn_action_frame, text="打开所在文件夹", command=self._open_result_dir, state=tk.DISABLED)
        self.btn_open_dir.pack(side=tk.LEFT, padx=5)

        self.btn_oam_view = ttk.Button(btn_action_frame, text="涡旋强度 / 相位 / 谱", command=self._show_oam_analysis, state=tk.DISABLED)
        self.btn_oam_view.pack(side=tk.LEFT, padx=5)

    def _on_beam_type_change(self, event=None):
        vortex = self.combo_beam_type.current() == 1
        if vortex == self._vortex_active:
            self._update_default_filename()
            return
        self._vortex_active = vortex
        self.combo_charge.configure(state='readonly' if vortex else 'disabled')
        self.ent_maxIt.configure(state='disabled' if vortex else 'normal')
        if vortex:
            self._plane_fields = [entry.get() for entry in (self.ent_theta, self.ent_phi, self.ent_weights)]
            self._plane_bit_index = self.combo_bit.current()
            self.combo_bit.current(0)
            for entry, value in zip((self.ent_theta, self.ent_phi, self.ent_weights), ('0', '0', '')):
                entry.configure(state='normal')
                entry.delete(0, tk.END)
                entry.insert(0, value)
                entry.configure(state='disabled')
            self.combo_bit.configure(state='disabled')
            self.ga_radio.configure(state='disabled')
            if self.algo_var.get() == 'ga_exact':
                self.algo_var.set('fast')
        else:
            self.combo_bit.configure(state='readonly')
            self.combo_bit.current(getattr(self, '_plane_bit_index', 0))
            self.ga_radio.configure(state='normal')
            values = getattr(self, '_plane_fields', ['30', '150', ''])
            for entry, value in zip((self.ent_theta, self.ent_phi, self.ent_weights), values):
                entry.configure(state='normal')
                entry.delete(0, tk.END)
                entry.insert(0, value)
        self._update_default_filename()

    def _update_default_filename(self):
        N = self.ent_N.get().strip() or "36"
        bit = "2" if "2" in self.combo_bit.get() else "1"
        theta_clean = self.ent_theta.get().replace(",", " ").strip()
        phi_clean = self.ent_phi.get().replace(",", " ").strip()
        time_str = datetime.datetime.now().strftime("%Y%m%d%H%M")
        
        # 默认保存在当前目录下的 output/ 目录
        out_dir = str(PROJECT_ROOT / 'output')
        os.makedirs(out_dir, exist_ok=True)
        fname = f"{time_str} {N} {bit} bit theta {theta_clean} phi {phi_clean}.txt"
        if self.combo_beam_type.current() == 1:
            fname = f"{time_str} {N} 2bit OAM_l{self.combo_charge.get()}.txt"
        self.file_path_var.set(os.path.join(out_dir, fname))

    def _choose_save_path(self):
        fpath = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=os.path.dirname(self.file_path_var.get() or "."),
            initialfile=os.path.basename(self.file_path_var.get() or "result.txt")
        )
        if fpath:
            self.file_path_var.set(fpath)

    def _parse_list(self, s):
        parts = s.replace(",", " ").split()
        return [float(p) for p in parts]

    def _run_optimization(self):
        # 收集输入并做校验
        try:
            freq = float(self.ent_freq.get().strip())
            period = float(self.ent_period.get().strip())
            N = int(self.ent_N.get().strip())
            if not np.isfinite(freq) or freq <= 0 or not np.isfinite(period) or period <= 0:
                raise ValueError("频率和周期必须为有限正数！")
            if N <= 0:
                raise ValueError("阵列尺寸必须为正整数！")
            nbit = 2 if "2" in self.combo_bit.get() else 1
            beam_type = 'vortex' if self.combo_beam_type.current() == 1 else 'plane'
            oam_charge = int(self.combo_charge.get()) if beam_type == 'vortex' else 1
            theta_list = self._parse_list(self.ent_theta.get().strip())
            phi_list = self._parse_list(self.ent_phi.get().strip())
            if len(theta_list) != len(phi_list):
                raise ValueError("俯仰角 θ 与方位角 φ 的数量不一致！")
            if len(theta_list) == 0:
                raise ValueError("请至少提供一个目标角度！")
            if not np.isfinite(theta_list).all() or not np.isfinite(phi_list).all():
                raise ValueError("角度必须为有限数值！")
            if any(th < 0 or th > 90 for th in theta_list):
                raise ValueError("俯仰角必须在 0 到 90 度之间！")
                
            weights_str = self.ent_weights.get().strip()
            weights = self._parse_list(weights_str) if weights_str else None
            if weights and len(weights) != len(theta_list):
                raise ValueError("波束权重数量必须与角度数量相同！")
            if weights is not None and (not np.isfinite(weights).all() or any(w < 0 for w in weights) or not any(w > 0 for w in weights)):
                raise ValueError("幅度权重必须有限、非负，且至少一个大于 0！")
                
            save_path = self.file_path_var.get().strip()
            if not save_path:
                self._update_default_filename()
                save_path = self.file_path_var.get().strip()
                
            method = self.algo_var.get()
            solver_options = {}
            if method == 'ga_exact':
                max_it = int(self.ent_maxIt.get().strip() or "4000")
                if max_it <= 0:
                    raise ValueError("GA 迭代次数必须大于 0！")
                solver_options['maxIt'] = max_it
        except Exception as err:
            messagebox.showerror("参数错误", f"输入参数格式有误：\n{err}")
            return
            
        # 禁用按钮并显示计算中
        self.btn_run.config(state=tk.DISABLED, text="正在优化计算中...")
        self.lbl_time.config(text="求解中...")
        self.update_idletasks()
        
        # 启动线程执行求解，避免界面假死
        def worker():
            try:
                res = design_metasurface(
                    frequency_thz=freq,
                    period_um=period,
                    theta=theta_list,
                    phi=phi_list,
                    N=N,
                    nbit=nbit,
                    method=method,
                    weights=weights,
                    output_path=save_path,
                    beam_type=beam_type,
                    oam_charge=oam_charge,
                    **solver_options
                )
                self.after(0, lambda: self._on_solve_finished(res, save_path, nbit, N))
            except Exception as e:
                self.after(0, lambda err_msg=str(e): self._on_solve_error(err_msg))
                
        threading.Thread(target=worker, daemon=True).start()

    def _on_solve_finished(self, res, save_path, nbit, N):
        self.btn_run.config(state=tk.NORMAL, text="▶ 立即生成编码矩阵")
        self.current_B = res['B']
        self.current_info = res
        self.current_save_path = save_path
        
        elapsed_ms = res['elapsed_sec'] * 1000.0
        self.lbl_time.config(text=f"耗时: {elapsed_ms:.2f} ms")
        if 'oam_analysis' in res:
            self.lbl_time.config(text=f"求解: {elapsed_ms:.0f} ms | 含评价导出: {res['total_elapsed_sec']:.2f} s")
        self.lbl_fitness.config(text=f"适应度: {res['fitness']:.2f}")
        self.lbl_evals.config(text=f"评价次数: {res['evals']}")
        
        # 统计分布
        vals, counts = np.unique(self.current_B, return_counts=True)
        dist_str = " | ".join([f"状态{v}: {c} ({c/(N*N)*100:.1f}%)" for v, c in zip(vals, counts)])
        self.info_lbl.config(text=f"文件已保存: {save_path}\n分布: {dist_str}")
        if 'oam_analysis' in res:
            metrics = res['oam_analysis']['metrics']
            self.lbl_fitness.config(text=f"OAM目标: {res['objective']:.5f}")
            wide = metrics['wide_angle']
            self.info_lbl.config(text=f"文件已保存: {save_path}\n分布: {dist_str}\n"
                f"目标 l={metrics['target_charge']:+d} | R={metrics['uv_radius']:.3f}圆盘内 {metrics['target_mode_fraction']:.1%}"
                f" | R={wide['uv_radius']:.2f}圆盘内 {wide['target_mode_fraction']:.1%}\n"
                f"环上绕转 {metrics['ring_winding']} | 轴心/峰值 {metrics['axis_to_peak_db']:.1f} dB\n"
                "绕转仅表示环内总荷；未验证居中、未分裂奇点。JSON / NPZ / PNG 已保存。")
            self.btn_oam_view.config(state=tk.NORMAL)
        else:
            self.btn_oam_view.config(state=tk.DISABLED)
        
        self.btn_open_file.config(state=tk.NORMAL)
        self.btn_open_dir.config(state=tk.NORMAL)
        
        # 绘制热力图
        self._render_matrix(self.current_B, nbit)

    def _on_solve_error(self, err_msg):
        self.btn_run.config(state=tk.NORMAL, text="▶ 立即生成编码矩阵")
        messagebox.showerror("运行失败", f"优化计算出现异常：\n{err_msg}")

    def _show_oam_analysis(self):
        if self.current_B is not None and 'oam_analysis' in self.current_info:
            from oam_view import show_oam_window
            show_oam_window(self, self.current_B, self.current_info['oam_analysis'])

    def _render_matrix(self, B, nbit):
        # 调整画布以适应阵列
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw < 50 or ch < 50:
            cw, ch = 400, 400
            
        M, N = B.shape
        side = min(cw - 40, ch - 40)
        cell_size = max(1, side // N)
        actual_w = cell_size * N
        actual_h = cell_size * M
        
        offset_x = (cw - actual_w) // 2
        offset_y = (ch - actual_h) // 2
        
        self.canvas.delete("all")
        
        # 经典超表面配色调色板 (Jet / Custom)
        if nbit == 1:
            palette = {0: "#1A5276", 1: "#E74C3C"} # 蓝 / 红
        else:
            palette = {0: "#1B4F72", 1: "#2E86C1", 2: "#F39C12", 3: "#C0392B"} # 深蓝, 浅蓝, 橙黄, 深红
            
        # 快速图像合成 (利用 PIL 生成再贴图，提高绘制速度)
        rgb_img = np.zeros((M, N, 3), dtype=np.uint8)
        color_map_rgb = {
            0: [27, 79, 114],
            1: [46, 134, 193],
            2: [243, 156, 18],
            3: [192, 57, 43]
        } if nbit == 2 else {
            0: [26, 82, 118],
            1: [231, 76, 60]
        }
        
        for k, rgb in color_map_rgb.items():
            rgb_img[B == k] = rgb
            
        pil_img = Image.fromarray(rgb_img, mode='RGB')
        pil_img = pil_img.resize((actual_w, actual_h), Image.Resampling.NEAREST)
        self.tk_img = ImageTk.PhotoImage(pil_img)
        
        self.canvas.create_image(offset_x, offset_y, anchor=tk.NW, image=self.tk_img)
        # 外边框
        self.canvas.create_rectangle(offset_x, offset_y, offset_x + actual_w, offset_y + actual_h, outline="#333333", width=2)

    def _open_result_file(self):
        if hasattr(self, 'current_save_path') and os.path.exists(self.current_save_path):
            os.system(f'notepad.exe "{self.current_save_path}"')

    def _open_result_dir(self):
        if hasattr(self, 'current_save_path'):
            folder = os.path.dirname(os.path.abspath(self.current_save_path))
            os.system(f'explorer.exe "{folder}"')

if __name__ == '__main__':
    app = MetasurfaceApp()
    app.mainloop()
