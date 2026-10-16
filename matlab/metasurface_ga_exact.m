function [best_B, best_fitness, history] = metasurface_ga_exact(varargin)
% METASURFACE_GA_EXACT 超表面离散编码的遗传算法求解器 (MATLAB 实现)
% 使用锦标赛选择、双点交叉、单点变异和精英筛选。
%
% 输入参数 (键值对或使用默认值):
%   'N'        : 阵列边长，默认 36 (阵列大小 N x N)
%   'nbit'     : 编码比特数，默认 2 (1-bit 或 2-bit)
%   'theta'    : 目标俯仰角列表 (度)，默认 [35, 45, 55]
%   'phi'      : 目标方位角列表 (度)，默认 [45, 135, 90]
%   'fre'      : 工作频率 (Hz)，默认 1.7e12 (1.7 THz)
%   'D'        : 单元周期 (m)，默认 70e-6 (70 um)
%   'weights'  : 各目标方向的复场幅度权重，默认全 1
%   'nPop'     : 种群大小，默认 50
%   'maxIt'    : 最大迭代次数，默认 4000
%   'nPc'      : 交叉率，默认 0.9
%   'nMu'      : 变异率，默认 0.8
%   'seed'     : 随机数种子，默认 42 (设为 [] 则不固定)
%   'save_path': 保存矩阵路径 (txt 或 mat)，默认 '' (不保存)
%   'plot_progress': 是否在最后绘图，默认 false

    p = inputParser;
    addParameter(p, 'N', 36, @isnumeric);
    addParameter(p, 'nbit', 2, @isnumeric);
    addParameter(p, 'theta', [35, 45, 55], @isnumeric);
    addParameter(p, 'phi', [45, 135, 90], @isnumeric);
    addParameter(p, 'fre', 1.7e12, @isnumeric);
    addParameter(p, 'D', 70e-6, @isnumeric);
    addParameter(p, 'weights', [], @isnumeric);
    addParameter(p, 'nPop', 50, @isnumeric);
    addParameter(p, 'maxIt', 4000, @isnumeric);
    addParameter(p, 'nPc', 0.9, @isnumeric);
    addParameter(p, 'nMu', 0.8, @isnumeric);
    addParameter(p, 'seed', 42);
    addParameter(p, 'save_path', '', @ischar);
    addParameter(p, 'plot_progress', false, @islogical);
    parse(p, varargin{:});
    opts = p.Results;

    validateattributes(opts.N, {'numeric'}, {'scalar', 'real', 'finite', 'integer', '>=', 2});
    validateattributes(opts.nbit, {'numeric'}, {'scalar', 'real', 'finite', 'integer'});
    assert(ismember(opts.nbit, [1, 2]), 'nbit必须为1或2。');
    validateattributes(opts.fre, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});
    validateattributes(opts.D, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});
    validateattributes(opts.theta, {'numeric'}, {'vector', 'real', 'finite', 'nonempty', '>=', 0, '<=', 90});
    validateattributes(opts.phi, {'numeric'}, {'vector', 'real', 'finite', 'nonempty'});
    assert(numel(opts.theta) == numel(opts.phi), 'theta与phi的数量必须相同。');
    validateattributes(opts.nPop, {'numeric'}, {'scalar', 'real', 'finite', 'integer', '>=', 2});
    validateattributes(opts.maxIt, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'});
    validateattributes(opts.nPc, {'numeric'}, {'scalar', 'real', 'finite', '>=', 0, '<=', 1});
    validateattributes(opts.nMu, {'numeric'}, {'scalar', 'real', 'finite', '>=', 0, '<=', 1});
    if ~isempty(opts.weights)
        validateattributes(opts.weights, {'numeric'}, {'vector', 'real', 'finite', 'nonnegative'});
        assert(numel(opts.weights) == numel(opts.theta) && any(opts.weights > 0), ...
            '幅度权重必须匹配波束数量，且至少一项大于0。');
    end

    if ~isempty(opts.seed)
        rng(opts.seed);
    end

    N = opts.N;
    nbit = opts.nbit;
    theta = opts.theta(:)';
    phi = opts.phi(:)';
    num_beams = length(theta);
    weights = opts.weights;
    if isempty(weights)
        weights = ones(1, num_beams);
    else
        weights = weights(:)';
    end

    % 1. 物理常数与导向矩阵预计算 (仅执行 1 次)
    ev0 = 8.854187817e-12;
    uv0 = 4 * pi * 1e-7;
    c0 = 1 / sqrt(ev0 * uv0);
    w = opts.fre * (2 * pi);
    k = w / c0;
    D = opts.D;

    % 坐标网格 (m-1/2, n-1/2)
    m_vec = (1:N) - 0.5;
    n_vec = (1:N) - 0.5;
    [Y_grid, X_grid] = meshgrid(n_vec, m_vec); % X 对应行 m, Y 对应列 n

    theta_rad = theta * pi / 180;
    phi_rad = phi * pi / 180;

    W = zeros(N, N);
    for kk = 1:num_beams
        u_k = sin(theta_rad(kk)) * cos(phi_rad(kk));
        v_k = sin(theta_rad(kk)) * sin(phi_rad(kk));
        psi_k = k * D * (X_grid * u_k + Y_grid * v_k);
        W = W + weights(kk) * exp(-1i * psi_k);
    end

    % 离散状态查找表
    num_states = 2^nbit;
    state_angles = (0:(num_states - 1)) * (pi / (2^(nbit - 1)));
    lookup_exp = exp(-1i * state_angles);

    % GA 参数
    nPop = opts.nPop;
    maxIt = opts.maxIt;
    nC = round(nPop * opts.nPc / 2) * 2;
    nMu = opts.nMu;
    nGenes = N * N;

    % 种群预分配 (扁平化存储以最大化向量化性能)
    Parent_B = randi([0, num_states - 1], nPop, nGenes);
    Parent_E = zeros(nPop, 1);
    Parent_y = zeros(nPop, 1);

    W_flat = W(:); % 列优先展平

    % 初始评价
    for h = 1:nPop
        exp_fa = lookup_exp(Parent_B(h, :) + 1);
        Parent_E(h) = sum(exp_fa(:) .* W_flat);
        Parent_y(h) = -abs(Parent_E(h));
    end

    history = zeros(1, maxIt);
    Offspring_B = zeros(nC, nGenes);
    Offspring_E = zeros(nC, 1);
    Offspring_y = zeros(nC, 1);

    % 主循环 (彻底移除循环内部绘图)
    for It = 1:maxIt
        % 锦标赛选择与双点交叉
        for j = 1:(nC / 2)
            % 选择 parent 1
            idx_a = randperm(nPop, 2);
            if Parent_y(idx_a(1)) <= Parent_y(idx_a(2))
                p1_idx = idx_a(1);
            else
                p1_idx = idx_a(2);
            end

            % 选择 parent 2
            idx_b = randperm(nPop, 2);
            if Parent_y(idx_b(1)) <= Parent_y(idx_b(2))
                p2_idx = idx_b(1);
            else
                p2_idx = idx_b(2);
            end

            p1 = Parent_B(p1_idx, :);
            p2 = Parent_B(p2_idx, :);

            % 在两个交叉位置之间交换父代编码
            pts = sort(randperm(nGenes - 1, 2));
            pt1 = pts(1);
            pt2 = pts(2);

            c1 = [p1(1:pt1), p2(pt1+1:pt2), p1(pt2+1:end)];
            c2 = [p2(1:pt1), p1(pt1+1:pt2), p2(pt2+1:end)];

            Offspring_B(2 * j - 1, :) = c1;
            Offspring_B(2 * j, :) = c2;
        end

        % 计算交叉后电场
        for k_idx = 1:nC
            exp_fa = lookup_exp(Offspring_B(k_idx, :) + 1);
            Offspring_E(k_idx) = sum(exp_fa(:) .* W_flat);
        end

        % 按概率 nMu 对一个随机阵元进行状态变异
        for k_idx = 1:nC
            if rand <= nMu
                s = randi([1, nGenes]);
                old_val = Offspring_B(k_idx, s);
                new_val = randi([0, num_states - 1]);
                if new_val ~= old_val
                    % O(1) 增量精确更新
                    delta = (lookup_exp(new_val + 1) - lookup_exp(old_val + 1)) * W_flat(s);
                    Offspring_E(k_idx) = Offspring_E(k_idx) + delta;
                    Offspring_B(k_idx, s) = new_val;
                end
            end
            Offspring_y(k_idx) = -abs(Offspring_E(k_idx));
        end

        % 精英筛选合并
        all_B = [Parent_B; Offspring_B];
        all_E = [Parent_E; Offspring_E];
        all_y = [Parent_y; Offspring_y];

        [~, so] = sort(all_y, 'ascend');
        Parent_B = all_B(so(1:nPop), :);
        Parent_E = all_E(so(1:nPop));
        Parent_y = all_y(so(1:nPop));

        history(It) = Parent_y(1);
    end

    % 恢复为 N x N 矩阵
    best_B = reshape(Parent_B(1, :), N, N);
    best_fitness = Parent_y(1);

    % 保存结果
    if ~isempty(opts.save_path)
        save_dir = fileparts(opts.save_path);
        if ~isempty(save_dir) && ~isfolder(save_dir), mkdir(save_dir); end
        [~, ~, ext] = fileparts(opts.save_path);
        if strcmpi(ext, '.mat')
            save(opts.save_path, 'best_B', 'best_fitness', 'history');
        else
            writematrix(best_B, opts.save_path);
        end
    end

    if opts.plot_progress
        figure;
        plot(history, 'LineWidth', 1.5);
        grid on;
        xlabel('迭代次数');
        ylabel('适应度 f (=-|S|)');
        title(sprintf('GA 收敛曲线 (最优适应度: %.2f)', best_fitness));
    end
end
