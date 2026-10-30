function run_unmixing_scene2RGB(mixture_id)
% 高光谱遥感影像混合像元分解实验 —— scene2_RGB
% 数据：Zhao et al. 2019 实验室合成数据集，空间变体端元线性混合
% 方法：VCA-FCLS、NMF、SNMF
% 指标：重构RMSE、重构相对误差、端元SAD、丰度RMSE（vs真值）
% 用法：run_unmixing_scene2RGB(1)  ~  run_unmixing_scene2RGB(4)
if nargin < 1 || isempty(mixture_id)
    mixture_id = 1;               % 1~4 选择第几幅混合影像
end
clc; close all;

%% 配置
data_root  = "Unmixing_dataset/scene2_RGB/";
out_dir    = "output/";
P          = 2;          % 每幅混合影像含2种端元（与真值丰度2通道对应）
max_iter   = 800;        % NMF/稀疏NMF最大迭代
tol        = 1e-7;       % NMF收敛容差
lambda     = 0.05;       % 稀疏NMF正则系数
eps        = 1e-10;
rng(7);                  % 固定随机种子，保证可复现
if ~exist(out_dir,'dir'), mkdir(out_dir); end

%% 加载混合影像 
mix     = load(fullfile(data_root, sprintf('scene2_RGB_mixture%d.mat', mixture_id)));
fm      = fieldnames(mix);
Y_cube  = mix.(fm{1});
[H,W,L] = size(Y_cube);  N = H*W;
Y       = reshape(Y_cube, N, L);      % N x L，每行一个像素光谱

%% 加载真值丰度 
gt    = load(fullfile(data_root, sprintf('scene2_RGB_ground_truth_mixture%d.mat', mixture_id)));
fg    = fieldnames(gt);
A_gt  = reshape(gt.(fg{1}), N, P);    % N x P

%% 参考端元光谱（取各纯端元小块的空间平均）
ref_names = {'blue','green','red','white'};
E_ref = zeros(L, 4);
for i = 1:4
    e    = load(fullfile(data_root, sprintf('scene2_RGB_end_%s.mat', ref_names{i})));
    fe   = fieldnames(e);
    E_ref(:,i) = squeeze(mean(mean(e.(fe{1}),1),2));   % L x 1
end

%% VCA-FCLS
E_vca = vca(Y, P);                    % L x P 端元
A_vca = zeros(N,P);
for i = 1:N
    A_vca(i,:) = fcls(Y(i,:)', E_vca)';
end
recon_vca = A_vca * E_vca';
rmse_vca  = sqrt(mean((Y(:)-recon_vca(:)).^2));
rel_vca   = norm(Y-recon_vca,'fro') / norm(Y,'fro');

%% NMF
A_nmf = rand(N,P)+0.1;  W_nmf = rand(L,P)+0.1;
err_old = inf;
for it = 1:max_iter
    W_nmf = W_nmf .* (Y'*A_nmf) ./ (W_nmf*(A_nmf'*A_nmf)+eps);
    A_nmf = A_nmf .* (Y*W_nmf)  ./ (A_nmf*(W_nmf'*W_nmf)+eps);
    if mod(it,20)==0
        e2 = norm(Y - A_nmf*W_nmf','fro') / norm(Y,'fro');
        if abs(e2-err_old) < tol, break; end
        err_old = e2;
    end
end
recon_nmf = A_nmf*W_nmf';
rmse_nmf  = sqrt(mean((Y(:)-recon_nmf(:)).^2));
rel_nmf   = norm(Y-recon_nmf,'fro') / norm(Y,'fro');

%% SNMF
A_s = rand(N,P)+0.1;  W_s = rand(L,P)+0.1;
for it = 1:max_iter
    W_s = W_s .* (Y'*A_s) ./ (W_s*(A_s'*A_s)+eps);
    A_s = A_s .* (Y*W_s)  ./ (A_s*(W_s'*W_s)+lambda+eps);
end
recon_s = A_s*W_s';
rmse_s  = sqrt(mean((Y(:)-recon_s(:)).^2));
rel_s   = norm(Y-recon_s,'fro') / norm(Y,'fro');

%% 定量评价
% SAD：估计端元到最近参考端元的光谱角（弧度）
sad_vca = mean_sad(E_ref, E_vca);
sad_nmf = mean_sad(E_ref, W_nmf);
sad_s   = mean_sad(E_ref, W_s);
% 丰度RMSE：估计丰度 vs 真值（两种列序取较小，因真值/估计通道顺序未知）
ar_vca = abundance_rmse(A_vca, A_gt);
ar_nmf = abundance_rmse(A_nmf, A_gt);
ar_s   = abundance_rmse(A_s,   A_gt);

%% 控制台输出
fprintf('scene2_RGB mixture%d：尺寸 %dx%dx%d，像素数 %d\n', mixture_id, H, W, L, N);
fprintf('%-10s %-12s %-12s %-10s %-12s\n','方法','重构RMSE','重构相对误差','SAD(rad)','丰度RMSE');
fprintf('%-10s %-12.5f %-12.4f %-10.5f %-12.5f\n','VCA+FCLS', rmse_vca, rel_vca, sad_vca, ar_vca);
fprintf('%-10s %-12.5f %-12.4f %-10.5f %-12.5f\n','NMF', rmse_nmf, rel_nmf, sad_nmf, ar_nmf);
fprintf('%-10s %-12.5f %-12.4f %-10.5f %-12.5f\n','sNMF', rmse_s, rel_s, sad_s, ar_s);

%% 可视化 
% 图1：VCA提取端元光谱 vs 匹配参考端元
figure('Visible','off','Position',[50 50 1100 460]);
for j = 1:P
    subplot(1,P,j);
    plot(E_ref, 'Color',[0.75 0.75 0.75],'LineWidth',0.6); hold on;
    [~, bi] = min_sad_idx(E_vca(:,j), E_ref);
    plot(E_ref(:,bi),'k--','LineWidth',1.2); hold on;
    plot(E_vca(:,j),'r','LineWidth',1.5);
    title(sprintf('VCA端元%d (匹配:%s)', j, ref_names{bi}));
    xlabel('波段'); ylabel('反射率');
end
saveas(gcf, fullfile(out_dir, sprintf('scene2RGB_m%d_endmembers.png', mixture_id)));

% 图2：VCA丰度图 vs 真值丰度图
figure('Visible','off','Position',[50 50 1000 620]);
for j = 1:P
    subplot(2,P,j);
    imagesc(reshape(A_vca(:,j),H,W)); axis image; colormap(parula); colorbar;
    title(sprintf('VCA丰度%d', j));
    subplot(2,P,j+P);
    imagesc(reshape(A_gt(:,j),H,W)); axis image; colormap(parula); colorbar;
    title(sprintf('真值丰度%d', j));
end
saveas(gcf, fullfile(out_dir, sprintf('scene2RGB_m%d_abundance.png', mixture_id)));

fprintf('图已保存：%s\n', fullfile(out_dir, sprintf('scene2RGB_m%d_endmembers.png', mixture_id)));
fprintf('图已保存：%s\n', fullfile(out_dir, sprintf('scene2RGB_m%d_abundance.png', mixture_id)));
end

%% ==================== 辅助函数 ====================

% VCA 端元提取
function [E, idx] = vca(Y, P)
    [N,L] = size(Y);
    R  = Y';                       % L x N
    Rm = mean(R,2);
    R0 = R - Rm;
    [U,~,~] = svd(R0,'econ');
    U  = U(:,1:P-1);               % L x (P-1) 主方向
    X  = [U'*R0; ones(1,N)];       % P x N 降维投影 + 常数偏移
    Aacc = zeros(P,P); idx = zeros(1,P); E = zeros(L,P);
    for i = 1:P
        w = randn(P,1);
        if i > 1
            f = w - Aacc(:,1:i-1)*(Aacc(:,1:i-1)\w);  % 正交于已找到端元
        else
            f = w;
        end
        f = f/norm(f);
        v = f'*X;
        [~,k] = max(abs(v));
        Aacc(:,i) = X(:,k); E(:,i) = R(:,k); idx(i) = k;
    end
end

% FCLS 单像素丰度：非负最小二乘后归一化到和为1（近似全约束）
function a = fcls(y, E)
    a = lsqnonneg(E, y);
    s = sum(a);
    if s < 1e-10
        a = ones(size(a))/numel(a);
    else
        a = a / s;
    end
end

% 平均SAD：每个估计端元到4个参考端元中最近者的光谱角均值
function sadm = mean_sad(E_ref, E_est)
    P = size(E_est,2);  sadv = zeros(1,P);
    for i = 1:P
        [~, bi] = min_sad_idx(E_est(:,i), E_ref);
        sadv(i) = acos(dot(E_est(:,i),E_ref(:,bi)) / (norm(E_est(:,i))*norm(E_ref(:,bi))+1e-10));
    end
    sadm = mean(sadv);
end

% 单条光谱到参考集的最小SAD及其索引
function [sadv, bi] = min_sad_idx(e, E_ref)
    n = size(E_ref,2); sadv = inf; bi = 1;
    for j = 1:n
        s = acos(dot(e,E_ref(:,j)) / (norm(e)*norm(E_ref(:,j))+1e-10));
        if s < sadv, sadv = s; bi = j; end
    end
end

% 丰度RMSE（两种列序取较小，处理通道顺序未知）
function r = abundance_rmse(A, G)
    r0 = sqrt(mean((A(:)-G(:)).^2));
    G2 = G(:, end:-1:1);
    r1 = sqrt(mean((A(:)-G2(:)).^2));
    r  = min(r0, r1);
end
