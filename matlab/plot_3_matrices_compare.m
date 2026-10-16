% plot_3_matrices_compare.m
% 对比用户生成的三个2-bit矩阵的二维采样远场，分别按各自峰值归一化。

project_root = fileparts(fileparts(mfilename('fullpath')));

f_fast = fullfile(project_root, 'output', 'fast1.txt');
f_ana  = fullfile(project_root, 'output', 'analytical.txt');
f_ga   = fullfile(project_root, 'output', 'ga1.txt');
if ~isfile(f_fast) || ~isfile(f_ana) || ~isfile(f_ga)
    error('请先在项目output目录生成相同物理条件下的三个待比较矩阵。');
end

B_fast = readmatrix(f_fast);
B_ana  = readmatrix(f_ana);
B_ga   = readmatrix(f_ga);
for matrix_cell = {B_fast, B_ana, B_ga}
    matrix = matrix_cell{1};
    if ~isequal(size(matrix), [36, 36]) || ~all(ismember(matrix(:), 0:3))
        error('当前绘图脚本需要三个36x36、2-bit整数编码矩阵。');
    end
end

N = 36;
fre = 1.7e12;
D = 70e-6;
c0 = 2.99792458e8;
lamda = c0 / fre;
k0 = 2 * pi / lamda;

th_deg = linspace(0, 60, 121);
ph_deg = linspace(0, 360, 181);
th_rad = deg2rad(th_deg);
ph_rad = deg2rad(ph_deg);

sin_th = sin(th_rad)';
cos_ph = cos(ph_rad);
sin_ph = sin(ph_rad);
u = sin_th * cos_ph;
v = sin_th * sin_ph;

[Y_g, X_g] = meshgrid((1:N)-0.5, (1:N)-0.5);

S_fast = calc_norm_ff(B_fast, X_g, Y_g, k0, D, u, v, th_rad, ph_rad);
S_ana  = calc_norm_ff(B_ana,  X_g, Y_g, k0, D, u, v, th_rad, ph_rad);
S_ga   = calc_norm_ff(B_ga,   X_g, Y_g, k0, D, u, v, th_rad, ph_rad);

fig = figure('Position', [100, 100, 1350, 420], 'Visible', 'off');

subplot(1, 3, 1);
contourf(u, v, S_fast, 25, 'LineStyle', 'none');
colorbar; colormap(jet); axis equal; axis tight;
title('1. 混合优化（按自身峰值归一化）', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('u = sin\theta cos\phi'); ylabel('v = sin\theta sin\phi');

subplot(1, 3, 2);
contourf(u, v, S_ana, 25, 'LineStyle', 'none');
colorbar; colormap(jet); axis equal; axis tight;
title('2. 解析综合（按自身峰值归一化）', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('u = sin\theta cos\phi'); ylabel('v = sin\theta sin\phi');

subplot(1, 3, 3);
contourf(u, v, S_ga, 25, 'LineStyle', 'none');
colorbar; colormap(jet); axis equal; axis tight;
title('3. 遗传算法（按自身峰值归一化）', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('u = sin\theta cos\phi'); ylabel('v = sin\theta sin\phi');

report_dir = fullfile(project_root, 'reports');
if ~isfolder(report_dir), mkdir(report_dir); end
report_path = fullfile(report_dir, 'user_3_matrices_compare.png');
saveas(fig, report_path);
fprintf('对比云图已保存至 %s\n', report_path);

function norm_S = calc_norm_ff(B, X_g, Y_g, k0, D, u, v, th_rad, ph_rad)
    fa = B * pi / 2;
    N = size(B, 1);
    S = zeros(length(th_rad), length(ph_rad));
    for m = 1:N
        for n = 1:N
            S = S + exp(-1i * (fa(m,n) + k0 * D * (X_g(m,n)*u + Y_g(m,n)*v)));
        end
    end
    norm_S = abs(S) / max(max(abs(S(:))), eps);
end
