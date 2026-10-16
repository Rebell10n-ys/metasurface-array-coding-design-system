function metasurface_gui()
% METASURFACE_GUI 超表面阵列编码设计系统 (MATLAB 原生图形界面版)
% 在 MATLAB 命令行直接输入 metasurface_gui 即可启动。

    addpath(fileparts(mfilename('fullpath')));
    
    % 创建主界面窗口
    fig = figure('Name', '超表面阵列编码设计系统 (MATLAB GUI)', ...
        'NumberTitle', 'off', ...
        'Position', [150, 150, 1000, 680], ...
        'MenuBar', 'none', ...
        'Resize', 'on', ...
        'Color', [0.94, 0.94, 0.94]);

    % 标题面板
    uicontrol(fig, 'Style', 'text', 'String', '超表面阵列离散编码设计系统', ...
        'FontName', 'Microsoft YaHei', 'FontSize', 15, 'FontWeight', 'bold', ...
        'HorizontalAlignment', 'left', ...
        'BackgroundColor', [0.94, 0.94, 0.94], ...
        'Position', [20, 635, 300, 30]);

    uicontrol(fig, 'Style', 'text', 'String', '太赫兹 1/2-bit 阵列波束赋形 | 毫秒级极速求解', ...
        'FontName', 'Microsoft YaHei', 'FontSize', 9, ...
        'ForegroundColor', [0.4, 0.4, 0.4], ...
        'HorizontalAlignment', 'left', ...
        'BackgroundColor', [0.94, 0.94, 0.94], ...
        'Position', [20, 615, 300, 20]);

    % 左侧控制面板
    pnl = uipanel(fig, 'Title', '设计需求与参数设置', ...
        'FontName', 'Microsoft YaHei', 'FontSize', 10, 'FontWeight', 'bold', ...
        'Position', [0.02, 0.05, 0.42, 0.86]);

    % 参数输入框
    y = 480;
    uicontrol(pnl, 'Style', 'text', 'String', 'GA 迭代代数:', 'Position', [15, y, 100, 20], 'HorizontalAlignment', 'left');
    edt_maxIt = uicontrol(pnl, 'Style', 'edit', 'String', '4000', 'Position', [120, y, 80, 25]);
    uicontrol(pnl, 'Style', 'text', 'String', '(仅限 ga_exact 生效，可自由修改)', 'Position', [210, y, 175, 20], ...
        'HorizontalAlignment', 'left', 'ForegroundColor', [0.4, 0.4, 0.4]);
    
    y = y - 45;
    uicontrol(pnl, 'Style', 'text', 'String', '工作频率 (THz):', 'Position', [15, y, 100, 20], 'HorizontalAlignment', 'left');
    edt_freq = uicontrol(pnl, 'Style', 'edit', 'String', '1.7', 'Position', [120, y, 80, 25]);

    uicontrol(pnl, 'Style', 'text', 'String', '单元周期 (μm):', 'Position', [215, y, 90, 20], 'HorizontalAlignment', 'left');
    edt_period = uicontrol(pnl, 'Style', 'edit', 'String', '70.0', 'Position', [310, y, 70, 25]);

    y = y - 40;
    uicontrol(pnl, 'Style', 'text', 'String', '阵列尺寸 (NxN):', 'Position', [15, y, 100, 20], 'HorizontalAlignment', 'left');
    edt_N = uicontrol(pnl, 'Style', 'edit', 'String', '36', 'Position', [120, y, 80, 25]);

    uicontrol(pnl, 'Style', 'text', 'String', '编码比特位数:', 'Position', [215, y, 90, 20], 'HorizontalAlignment', 'left');
    pop_bit = uicontrol(pnl, 'Style', 'popupmenu', 'String', {'2-bit (0,1,2,3)', '1-bit (0,1)'}, 'Position', [310, y, 70, 25]);

    y = y - 40;
    uicontrol(pnl, 'Style', 'text', 'String', '目标俯仰角 θ (°):', 'Position', [15, y, 100, 20], 'HorizontalAlignment', 'left');
    edt_theta = uicontrol(pnl, 'Style', 'edit', 'String', '30', 'Position', [120, y, 260, 25]);

    y = y - 40;
    uicontrol(pnl, 'Style', 'text', 'String', '目标方位角 φ (°):', 'Position', [15, y, 100, 20], 'HorizontalAlignment', 'left');
    edt_phi = uicontrol(pnl, 'Style', 'edit', 'String', '150', 'Position', [120, y, 260, 25]);

    y = y - 40;
    uicontrol(pnl, 'Style', 'text', 'String', '波束权重 (可选):', 'Position', [15, y, 100, 20], 'HorizontalAlignment', 'left');
    edt_weights = uicontrol(pnl, 'Style', 'edit', 'String', '', 'Position', [120, y, 260, 25]);

    y = y - 45;
    uicontrol(pnl, 'Style', 'text', 'String', '求解算法选择:', 'Position', [15, y, 100, 20], 'HorizontalAlignment', 'left');
    pop_algo = uicontrol(pnl, 'Style', 'popupmenu', ...
        'String', {'解析初始化与坐标下降 (fast)', '遗传算法搜索 (ga_exact)'}, ...
        'Position', [120, y, 260, 25]);

    y = y - 50;
    uicontrol(pnl, 'Style', 'text', 'String', '保存 TXT 文件路径:', 'Position', [15, y, 110, 20], 'HorizontalAlignment', 'left');
    edt_path = uicontrol(pnl, 'Style', 'edit', 'String', fullfile(pwd, 'output_array.txt'), 'Position', [15, y - 25, 290, 25]);
    btn_browse = uicontrol(pnl, 'Style', 'pushbutton', 'String', '浏览...', 'Position', [310, y - 25, 70, 25]);

    % 运行按钮
    btn_run = uicontrol(pnl, 'Style', 'pushbutton', 'String', '▶ 立即生成编码矩阵', ...
        'FontName', 'Microsoft YaHei', 'FontSize', 12, 'FontWeight', 'bold', ...
        'ForegroundColor', 'white', 'BackgroundColor', [0, 0.45, 0.74], ...
        'Position', [15, 20, 365, 45]);

    % 右侧展示面板
    pnl_right = uipanel(fig, 'Title', '编码排布可视化与指标', ...
        'FontName', 'Microsoft YaHei', 'FontSize', 10, 'FontWeight', 'bold', ...
        'Position', [0.46, 0.05, 0.52, 0.86]);

    lbl_status = uicontrol(pnl_right, 'Style', 'text', 'String', '就绪，请点击左侧【立即生成编码矩阵】', ...
        'FontName', 'Microsoft YaHei', 'FontSize', 10, ...
        'HorizontalAlignment', 'left', ...
        'Position', [20, 515, 460, 25]);

    ax = axes('Parent', pnl_right, 'Units', 'pixels', 'Position', [50, 80, 380, 380]);
    axis(ax, 'equal');
    box(ax, 'on');

    % 状态栏与操作按钮
    btn_open = uicontrol(pnl_right, 'Style', 'pushbutton', 'String', '打开生成文件', ...
        'Position', [50, 25, 110, 30], 'Enable', 'off');

    btn_folder = uicontrol(pnl_right, 'Style', 'pushbutton', 'String', '打开所在文件夹', ...
        'Position', [175, 25, 110, 30], 'Enable', 'off');

    set(btn_browse, 'Callback', @(~, ~) on_browse());
    set(btn_run, 'Callback', @(~, ~) on_run());
    set(btn_open, 'Callback', @(~, ~) on_open_file());
    set(btn_folder, 'Callback', @(~, ~) on_open_folder());

    current_file = '';

    function on_browse()
        [f, p] = uiputfile('*.txt', '保存编码矩阵', get(edt_path, 'String'));
        if f ~= 0
            set(edt_path, 'String', fullfile(p, f));
        end
    end

    function on_run()
        try
            set(btn_run, 'Enable', 'off', 'String', '正在优化计算中...');
            drawnow;
            
            freq_val = str2double(get(edt_freq, 'String')) * 1e12;
            period_val = str2double(get(edt_period, 'String')) * 1e-6;
            N_val = str2double(get(edt_N, 'String'));
            nbit_val = 3 - get(pop_bit, 'Value'); % 1 -> 2-bit, 2 -> 1-bit
            th_val = str2num(get(edt_theta, 'String')); %#ok<ST2NM>
            ph_val = str2num(get(edt_phi, 'String')); %#ok<ST2NM>
            w_str = get(edt_weights, 'String');
            if isempty(w_str)
                w_val = [];
            else
                w_val = str2num(w_str); %#ok<ST2NM>
            end
            
            maxIt_val = str2double(get(edt_maxIt, 'String'));
            if isnan(maxIt_val) || maxIt_val <= 0
                maxIt_val = 4000;
            end
            
            save_path = get(edt_path, 'String');
            algo_choice = get(pop_algo, 'Value');
            
            t0 = tic;
            if algo_choice == 1
                [B, f, t_sec] = metasurface_fast('N', N_val, 'nbit', nbit_val, ...
                    'theta', th_val, 'phi', ph_val, 'fre', freq_val, 'D', period_val, ...
                    'weights', w_val, 'save_path', save_path);
            else
                [B, f, ~] = metasurface_ga_exact('N', N_val, 'nbit', nbit_val, ...
                    'theta', th_val, 'phi', ph_val, 'fre', freq_val, 'D', period_val, ...
                    'weights', w_val, 'maxIt', maxIt_val, 'save_path', save_path);
                t_sec = toc(t0);
            end
            
            current_file = save_path;
            set(lbl_status, 'String', sprintf('计算完成！耗时: %.2f ms | 适应度: %.2f | 已保存 TXT', t_sec * 1000, f), ...
                'ForegroundColor', [0, 0.5, 0]);
            
            % 绘图
            axes(ax);
            imagesc(ax, 1:N_val, 1:N_val, B);
            axis(ax, 'equal'); axis(ax, 'tight');
            colorbar(ax);
            colormap(ax, jet);
            title(ax, sprintf('超表面阵列编码图 (%dx%d, %d-bit)', N_val, N_val, nbit_val), 'FontWeight', 'bold');
            
            set(btn_open, 'Enable', 'on');
            set(btn_folder, 'Enable', 'on');
            set(btn_run, 'Enable', 'on', 'String', '▶ 立即生成编码矩阵');
        catch err
            set(btn_run, 'Enable', 'on', 'String', '▶ 立即生成编码矩阵');
            errordlg(['运行出错: ' err.message], '错误');
        end
    end

    function on_open_file()
        if exist(current_file, 'file')
            winopen(current_file);
        end
    end

    function on_open_folder()
        if exist(current_file, 'file')
            p = fileparts(current_file);
            winopen(p);
        end
    end
end
