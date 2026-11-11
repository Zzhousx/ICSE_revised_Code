function [rankedIndices, tangentVectors, q_scores] = ...
        VoD_Sampler_ablation(data, alpha, window_size, step_size, variant)
    % variant
    
    if nargin < 5
        variant = 'Ours'; 
    end

    N       = length(data);
    D_base  = 18;
    D_aug   = D_base * 2;
    epsilon = 1e-5;
    feature_weights = [repmat(0.5,1,10), 3.0, 3.0, repmat(1.5,1,4), 0.5, 0.5];

    dim_vec = D_aug * (D_aug + 1) / 2;

    all_features = cat(1, data.Features);
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

    tangentVectors = zeros(N, dim_vec);
    pot_chaotic    = zeros(N, 1);
    pot_intensity  = zeros(N, 1);
    
    entropy_scores = zeros(N, 1); 

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
                Z = [X_norm(2:end,:), X_norm(1:end-1,:)];  
            else
                Z = zeros(0, D_aug);
            end

            if size(Z,1) < 2
                C         = eye(D_aug) * epsilon;
                z_mean_sq = 0;
            else
                z_mean    = mean(Z, 1);
                Z_c       = Z - z_mean;
                C         = (Z_c' * Z_c) / (size(Z,1)-1);
                z_mean_sq = sum(z_mean.^2);
            end
            
            C = C + epsilon * eye(D_aug);

            temp_chaotic(w) = trace(C);
            temp_intens(w)  = z_mean_sq;

            if strcmp(variant, 'w/o Riemannian')
                temp_S{w} = C; 
            else
                [U, Lam]   = eig(C, 'vector');
                d_log      = log(max(Lam, 1e-10));
                temp_S{w}  = U * diag(d_log) * U';
            end
        end

        [~, max_idx]    = max(temp_intens);
        pot_chaotic(i)  = temp_chaotic(max_idx);
        pot_intensity(i)= temp_intens(max_idx);
        
        S_best = temp_S{max_idx};

        if strcmp(variant, 'w/o Diversity')
            if strcmp(variant, 'w/o Riemannian') 
                entropy_scores(i) = sum(log(max(eig(S_best), 1e-10)));
            else
                entropy_scores(i) = trace(S_best);
            end
        end

        mask_d = logical(eye(D_aug));
        mask_u = triu(true(D_aug), 1);
        tangentVectors(i,:) = [S_best(mask_d); S_best(mask_u)*sqrt(2)]';
    end

    vecNorms = sqrt(sum(tangentVectors.^2, 2));

    if strcmp(variant, 'w/o Quality')
        tangentVectors_eval = tangentVectors; 
    else
        tangentVectors_eval = tangentVectors ./ (vecNorms + 1e-9);
    end

    minmax = @(x) (x-min(x)) ./ (max(x)-min(x) + 1e-9);
    
    if strcmp(variant, 'w/o Quality')
        q_scores = ones(N, 1); 
    else
        q_scores = alpha * minmax(pot_chaotic) + (1-alpha) * minmax(pot_intensity) + 1e-4;
    end

    if strcmp(variant, 'w/o Diversity')
        [~, rankedIndices] = sort(entropy_scores, 'descend');
        
    else
        B = tangentVectors_eval .* q_scores;
        rankedIndices = feature_space_greedy_dpp(B, N, q_scores);
    end
end