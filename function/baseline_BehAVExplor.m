function ranked = baseline_BehAVExplor(data)
% =========================================================================
% Baseline: BehAVExplor (Offline Adapted Version)
% =========================================================================

    N = length(data);
    if N <= 1, ranked = (1:N)'; return; end
    w = 10;             
    K_abs = 30;         
    alpha = 0.2;        
    feat_cell = cell(N, 1);
    L = zeros(N, 1); 
    native_violation = zeros(N, 1);
    
    for i = 1:N
        X = data(i).Features;
        T = size(X, 1);
        
        if T < 1
           feat_cell{i} = zeros(1, 36);
           L(i) = 1;
           native_violation(i) = 0;
           continue;
        end
        kinematic_cols = [3, 4, 5, 6, 9, 10];
        kinematic_energy = vecnorm(X(:, kinematic_cols), 2, 2); 
        native_violation(i) = mean(kinematic_energy);
        num_win = ceil(T / w);
        win_feat = zeros(num_win, 36);
        for j = 1:num_win
            start_idx = (j-1)*w + 1;
            end_idx = min(j*w, T);
            win_data = X(start_idx:end_idx, :);
            
            win_feat(j, 1:18) = mean(win_data, 1);
            if size(win_data, 1) > 1
                win_feat(j, 19:36) = std(win_data, 0, 1);
            else
                win_feat(j, 19:36) = 0;
            end
        end
        feat_cell{i} = win_feat;
        L(i) = num_win;
    end
    
    all_win_feats = cell2mat(feat_cell);
    
    % Global Normalization
    mu = mean(all_win_feats, 1);
    sigma = std(all_win_feats, 0, 1) + 1e-8;
    all_win_feats = (all_win_feats - mu) ./ sigma;
    
    num_samples = size(all_win_feats, 1);
    sample_idx = randperm(num_samples, min(50000, num_samples));
    
    opts = statset('Display','off', 'MaxIter', 1000);
    [~, centroids] = kmeans(all_win_feats(sample_idx, :), K_abs, 'Options', opts, 'Replicates', 100);


    dist_mat = pdist2(all_win_feats, centroids);
    [~, state_assignments] = min(dist_mat, [], 2);
    
    L_max = max(L);
    H = zeros(N, L_max); 
    cur_idx = 1;
    for i = 1:N
        H(i, 1:L(i)) = state_assignments(cur_idx : cur_idx + L(i) - 1)';
        cur_idx = cur_idx + L(i);
    end
    
    vio_norm = (native_violation - min(native_violation)) ./ ...
               (max(native_violation) - min(native_violation) + 1e-9);
    
    ranked = zeros(N, 1);
    unranked = (1:N)';
    min_dist = inf(N, 1); 
    
    [~, best_idx] = max(vio_norm);
    c = best_idx;
    
    for step = 1:N
        ranked(step) = c;
        unranked(unranked == c) = [];
        
        if isempty(unranked), break; end
        
        H_c = H(c, :);
        L_c = L(c);
        
        diff_mask = (H(unranked, :) ~= H_c) & (H(unranked, :) ~= 0) & (H_c ~= 0);
        hamming_dist = sum(diff_mask, 2);
        
        max_len = max(L(unranked), L_c);
        len_diff = abs(L(unranked) - L_c);
        
        dists = (hamming_dist + len_diff) ./ max_len;
        min_dist(unranked) = min(min_dist(unranked), dists);
        
        scores = alpha * vio_norm(unranked) + (1 - alpha) * min_dist(unranked);
        
        [~, max_s] = max(scores);
        c = unranked(max_s);
    end
end