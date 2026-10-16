function [best_B, best_fitness, t_sec] = metasurface_fast(varargin)
% METASURFACE_FAST 基于解析相位共轭与贪心坐标下降的超快速超表面编码设计 (MATLAB 实现)
% 策略 (Type B):
%   1. 解析相位共轭直接综合 (Analytical Phase Matching)
%   2. 增量局部坐标下降精修 (Greedy Coordinate Descent / ICM)
% 以目标方向复场的加权相干和幅值为目标进行离散局部搜索。
%
% 输入参数 (键值对):
%   'N'        : 阵列边长，默认 36 (N x N)
%   'nbit'     : 编码比特数，默认 2 (1 或 2)
%   'theta'    : 目标俯仰角列表 (度)，默认 [35, 45, 55]
%   'phi'      : 目标方位角列表 (度)，默认 [45, 135, 90]
%   'fre'      : 工作频率 (Hz)，默认 1.7e12 (1.7 THz)
%   'D'        : 单元周期 (m)，默认 70e-6 (70 um)
%   'weights'  : 目标波束权重，默认全 1
%   'maxPasses': 局部搜索轮数，默认 3
%   'save_path': 输出文件路径，默认 ''

    p = inputParser;
    addParameter(p, 'N', 36, @isnumeric);
    addParameter(p, 'nbit', 2, @isnumeric);
    addParameter(p, 'theta', [35, 45, 55], @isnumeric);
    addParameter(p, 'phi', [45, 135, 90], @isnumeric);
    addParameter(p, 'fre', 1.7e12, @isnumeric);
    addParameter(p, 'D', 70e-6, @isnumeric);
    addParameter(p, 'weights', [], @isnumeric);
    addParameter(p, 'maxPasses', 3, @isnumeric);
    addParameter(p, 'save_path', '', @ischar);
    parse(p, varargin{:});
    opts = p.Results;

    validateattributes(opts.N, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'positive'});
    validateattributes(opts.nbit, {'numeric'}, {'scalar', 'real', 'finite', 'integer'});
    assert(ismember(opts.nbit, [1, 2]), 'nbit必须为1或2。');
    validateattributes(opts.fre, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});
    validateattributes(opts.D, {'numeric'}, {'scalar', 'real', 'finite', 'positive'});
    validateattributes(opts.theta, {'numeric'}, {'vector', 'real', 'finite', 'nonempty', '>=', 0, '<=', 90});
    validateattributes(opts.phi, {'numeric'}, {'vector', 'real', 'finite', 'nonempty'});
    assert(numel(opts.theta) == numel(opts.phi), 'theta与phi的数量必须相同。');
    validateattributes(opts.maxPasses, {'numeric'}, {'scalar', 'real', 'finite', 'integer', 'nonnegative'});
    if ~isempty(opts.weights)
        validateattributes(opts.weights, {'numeric'}, {'vector', 'real', 'finite', 'nonnegative'});
        assert(numel(opts.weights) == numel(opts.theta) && any(opts.weights > 0), ...
            '幅度权重必须匹配波束数量，且至少一项大于0。');
    end

    t_start = tic;

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

    % 1. 预计算导向矩阵 W
    ev0 = 8.854187817e-12;
    uv0 = 4 * pi * 1e-7;
    c0 = 1 / sqrt(ev0 * uv0);
    w = opts.fre * (2 * pi);
    k = w / c0;
    D = opts.D;

    m_vec = (1:N) - 0.5;
    n_vec = (1:N) - 0.5;
    [Y_grid, X_grid] = meshgrid(n_vec, m_vec);

    theta_rad = theta * pi / 180;
    phi_rad = phi * pi / 180;

    W = zeros(N, N);
    for kk = 1:num_beams
        u_k = sin(theta_rad(kk)) * cos(phi_rad(kk));
        v_k = sin(theta_rad(kk)) * sin(phi_rad(kk));
        psi_k = k * D * (X_grid * u_k + Y_grid * v_k);
        W = W + weights(kk) * exp(-1i * psi_k);
    end

    % 2. 策略 B1: 解析相位共轭直接综合与离散量化
    num_states = 2^nbit;
    step = 2 * pi / num_states;
    target_continuous_phase = mod(angle(W), 2 * pi);
    B = mod(round(target_continuous_phase / step), num_states);

    state_angles = (0:(num_states - 1)) * (pi / (2^(nbit - 1)));
    lookup_exp = exp(-1i * state_angles);

    % 当前电场与适应度
    E_curr = sum(sum(lookup_exp(B + 1) .* W));
    curr_score = abs(E_curr);

    % 3. 策略 B2: 增量局部坐标下降精修
    maxPasses = opts.maxPasses;
    coords = randperm(N * N);

    for pass = 1:maxPasses
        improved = false;
        for idx = 1:(N * N)
            loc = coords(idx);
            [m, n] = ind2sub([N, N], loc);
            old_val = B(m, n);
            best_val = old_val;
            best_score = curr_score;
            best_E = E_curr;

            W_mn = W(m, n);
            old_exp = lookup_exp(old_val + 1);

            for v = 0:(num_states - 1)
                if v == old_val
                    continue;
                end
                cand_E = E_curr + (lookup_exp(v + 1) - old_exp) * W_mn;
                cand_score = abs(cand_E);
                if cand_score > best_score
                    best_score = cand_score;
                    best_val = v;
                    best_E = cand_E;
                end
            end

            if best_val ~= old_val
                B(m, n) = best_val;
                E_curr = best_E;
                curr_score = best_score;
                improved = true;
            end
        end
        if ~improved
            break;
        end
    end

    best_B = B;
    best_fitness = -curr_score;
    t_sec = toc(t_start);

    % 保存结果
    if ~isempty(opts.save_path)
        save_dir = fileparts(opts.save_path);
        if ~isempty(save_dir) && ~isfolder(save_dir), mkdir(save_dir); end
        [~, ~, ext] = fileparts(opts.save_path);
        if strcmpi(ext, '.mat')
            save(opts.save_path, 'best_B', 'best_fitness', 't_sec');
        else
            writematrix(best_B, opts.save_path);
        end
    end
end
