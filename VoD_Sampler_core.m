function [rankedIndices, tangentVectors, q_scores] = ...
        VoD_Sampler_core(data, alpha, window_size, step_size)

    N       = length(data);
    D_base  = 18;
    D_aug   = D_base * 2;
    dim_vec = D_aug * (D_aug + 1) / 2;
    epsilon = 1e-5;
    dt      = 0.5; 
    feature_weights = [repmat(0.5,1,10), 3.0, 3.0, repmat(1.5,1,4), 0.5, 0.5];

    % 全局标准化参数
    all_features = cat(1, data.Features);

    % ghost NPC 清零（与原始代码一致）
    ghost_mask = all_features(:,11) < 0.0183;
    all_features(ghost_mask, 11)    = 0;
    all_features(ghost_mask, 12)    = 0;
    all_features(ghost_mask, 13:14) = 0;
    all_features(ghost_mask, 15:16) = 0;

    idx_counter = 1;
    for i = 1:N
        T_len = size(data(i).Features, 1);
        data(i).Features = all_features(idx_counter : idx_counter+T_len-1, :);
        idx_counter = idx_counter + T_len;
    end

    global_mu    = mean(all_features);
    global_sigma = std(all_features) + 1e-8;
    clear all_features;

    % ------ Riemannian 嵌入 ------
    tangentVectors = zeros(N, dim_vec);
    pot_chaotic    = zeros(N, 1);
    pot_intensity  = zeros(N, 1);

    for i = 1:N
        raw_X = data(i).Features;
        T     = size(raw_X, 1);

        if T <= window_size
            wStarts = 1;  wEnds = T;
        else
            wStarts = 1 : step_size : (T - window_size + 1);
            wEnds   = wStarts + window_size - 1;
            if wEnds(end) < T
                wStarts(end+1) = T - window_size + 1;
                wEnds(end+1)   = T;
            end
        end

        nW           = length(wStarts);
        temp_chaotic = zeros(nW,1);
        temp_intens  = zeros(nW,1);
        temp_S       = cell(nW,1);

        for w = 1:nW
            X_win  = raw_X(wStarts(w):wEnds(w), :);
            X_norm = (X_win - global_mu) ./ global_sigma .* feature_weights;

            if size(X_norm,1) > 1
                Z = [X_norm(2:end,:), X_norm(1:end-1,:)];  % 因果增广
            else
                Z = zeros(0, D_aug);
            end

            if size(Z,1) < 2
                C       = eye(D_aug) * epsilon;
                z_mean_sq = 0;
            else
                z_mean    = mean(Z);
                Z_c       = Z - z_mean;
                C         = (Z_c' * Z_c) / (size(Z,1)-1);
                z_mean_sq = sum(z_mean.^2);
            end
            C = C + epsilon * eye(D_aug);

            temp_chaotic(w) = trace(C);
            temp_intens(w)  = z_mean_sq;

            [U, Lam]   = eig(C, 'vector');
            d_log      = log(max(Lam, 1e-10));
            temp_S{w}  = U * diag(d_log) * U';
        end

        [~, max_idx]    = max(temp_intens);
        pot_chaotic(i)  = temp_chaotic(max_idx);
        pot_intensity(i)= temp_intens(max_idx);

        S_best = temp_S{max_idx};
        mask_d = logical(eye(D_aug));
        mask_u = triu(true(D_aug), 1);
        tangentVectors(i,:) = [S_best(mask_d); S_best(mask_u)*sqrt(2)]';
    end

    % 归一化切空间向量
    vecNorms       = sqrt(sum(tangentVectors.^2, 2));
    tangentVectors = tangentVectors ./ (vecNorms + 1e-9);

    % 质量得分
    minmax    = @(x) (x-min(x)) ./ (max(x)-min(x) + 1e-9);
    q_scores  = alpha * minmax(pot_chaotic) + (1-alpha) * minmax(pot_intensity) + 1e-4;

    % DPP 双空间推断 —— 调用你已有的函数
    B              = tangentVectors .* q_scores;
    rankedIndices  = feature_space_greedy_dpp(B, N, q_scores);
end