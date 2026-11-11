function best_ranked = baseline_SDC_GA(data, mode)
% =========================================================================
% BASELINE: SDC-Prioritizer
% =========================================================================

    N = length(data);
    if N <= 1, best_ranked = (1:N)'; return; end

    is_MO = strcmp(mode, 'MO');

    % --- GA Hyperparameters ---
    pop_size = 100;
    max_gen  = 4000;
    p_c      = 0.80;
    p_m      = 1 / N;

    X     = zeros(N, 54);
    costs = zeros(N, 1);
    
    for i = 1:N
        F = data(i).Features;
        costs(i) = data(i).Duration;
        
        T = size(F, 1);
        if T > 1
            feat_max  = max(abs(F), [], 1);
            feat_std  = std(F, 0, 1);
            term_idx  = max(1, T - 2);
            feat_term = mean(F(term_idx:T, :), 1);
            
            X(i, :) = [feat_max, feat_std, feat_term];
        else
            if T == 1
                X(i, :) = [abs(F), zeros(1, 18), F];
            end
        end
    end
    
  
    costs = max(costs, 1e-5);
    costs = (costs - min(costs)) ./ (max(costs) - min(costs) + 1e-8) + 1.0;

    X_std  = std(X, 0, 1) + 1e-8;
    X_norm = (X - mean(X, 1)) ./ X_std;

    native_risk = sum(X_norm.^2, 2);
    native_risk = (native_risk - min(native_risk)) ./ (max(native_risk) - min(native_risk) + 1e-8) + 0.5;

    dist_mat = single(pdist2(X_norm, X_norm, 'cosine'));
    dist_mat = dist_mat .* single(native_risk'); 
    pop      = zeros(pop_size, N, 'int32');   
    n_greedy = round(pop_size * 0.30);

    for g = 1:n_greedy
        pop(g, :) = greedy_init_fast(dist_mat, costs, N);
    end
    for i = (n_greedy+1):pop_size
        pop(i, :) = int32(randperm(N));
    end

    pos_w  = single(1 ./ (2:N));      
    cost_w = single(1 ./ costs');     

    offspring = zeros(pop_size, N, 'int32');

    for gen = 1:max_gen
        % Vectorized Evaluation
        if ~is_MO
            f1 = eval_SO_vec(pop, dist_mat, cost_w, pos_w, N, pop_size);
        else
            [f1, f2] = eval_MO_vec(pop, dist_mat, cost_w, pos_w, N, pop_size);
            [ranks, crowding] = nsga2_sort_vec(f1, f2, pop_size);
        end

        for i = 1:2:pop_size
            if ~is_MO
                p1 = roulette_select(f1); p2 = roulette_select(f1);
            else
                p1 = binary_tournament(ranks, crowding);
                p2 = binary_tournament(ranks, crowding);
            end

            par1 = double(pop(p1, :)); par2 = double(pop(p2, :));

            if rand() < p_c
                [c1, c2] = pmx_crossover(par1, par2, N);
            else
                c1 = par1; c2 = par2;
            end

            c1 = mutate(c1, p_m, N); c2 = mutate(c2, p_m, N);

            offspring(i,   :) = int32(c1);
            if i+1 <= pop_size
                offspring(i+1, :) = int32(c2);
            end
        end

        combined = [pop; offspring];   

        if ~is_MO
            cf1 = eval_SO_vec(combined, dist_mat, cost_w, pos_w, N, 2*pop_size);
            [~, sidx] = sort(cf1, 'descend');
            pop = combined(sidx(1:pop_size), :);
        else
            [cf1, cf2] = eval_MO_vec(combined, dist_mat, cost_w, pos_w, N, 2*pop_size);
            [r2, cd2]  = nsga2_sort_vec(cf1, cf2, 2*pop_size);
            [~, sidx]  = sortrows([r2, -cd2]);
            pop = combined(sidx(1:pop_size), :);
        end
    end

    if ~is_MO
        final_f1 = eval_SO_vec(pop, dist_mat, cost_w, pos_w, N, pop_size);
        [~, best] = max(final_f1);
        best_ranked = double(pop(best, :))';
    else
        [f1_fin, f2_fin] = eval_MO_vec(pop, dist_mat, cost_w, pos_w, N, pop_size);
        [r_fin, ~]       = nsga2_sort_vec(f1_fin, f2_fin, pop_size);
        pf               = find(r_fin == 1);

        U1 = max(f1_fin(pf)); U2 = min(f2_fin(pf));
        range_f1 = max(f1_fin(pf)) - min(f1_fin(pf)) + 1e-9;
        range_f2 = max(f2_fin(pf)) - min(f2_fin(pf)) + 1e-9;

        d2u = ((f1_fin(pf) - U1) / range_f1).^2 + ((f2_fin(pf) - U2) / range_f2).^2;
        [~, ki]     = min(d2u);
        best_ranked = double(pop(pf(ki), :))';
    end
end

function f1 = eval_SO_vec(pop, dist_mat, cost_w, pos_w, N, P)
    from_idx = double(pop(:, 1:N-1));
    to_idx   = double(pop(:, 2:N));
    lin_idx = from_idx + (to_idx - 1) * N;   
    D = double(dist_mat(lin_idx));            
    W = cost_w(to_idx) .* pos_w;                      
    f1 = sum(D .* W, 2);                     
end

function [f1, f2] = eval_MO_vec(pop, dist_mat, cost_w, pos_w, N, P)
    from_idx = double(pop(:, 1:N-1));
    to_idx   = double(pop(:, 2:N));
    lin_idx = from_idx + (to_idx - 1) * N;
    D = double(dist_mat(lin_idx));           
    f1 = sum(D .* pos_w, 2);               
    costs_to = 1 ./ cost_w(to_idx);        
    costs_t1 = 1 ./ cost_w(double(pop(:,1)));  
    f2 = costs_t1(:) + sum(costs_to .* pos_w, 2); 
end

function [ranks, crowding] = nsga2_sort_vec(f1, f2, P)
    g1 = -f1(:); g2 = f2(:);   
    dg1 = g1 - g1'; dg2 = g2 - g2';   
    dom = (dg1 <= 0) & (dg2 <= 0) & ((dg1 < 0) | (dg2 < 0));
    dom(1:P+1:end) = false;  
    n_dominated = sum(dom, 1)';   
    ranks = zeros(P, 1); crowding = zeros(P, 1); unranked = true(P, 1); r = 1;

    while any(unranked)
        front_mask = unranked & (n_dominated == 0);
        if ~any(front_mask), ranks(unranked) = r; break; end
        front = find(front_mask); ranks(front) = r;
        L = length(front);
        if L <= 2
            crowding(front) = inf;
        else
            crowding(front) = crowding_distance_vec(g1(front), g2(front), L);
        end
        dominated_by_front = any(dom(front, :), 1)';  
        n_dominated(dominated_by_front & unranked) = n_dominated(dominated_by_front & unranked) - sum(dom(front, dominated_by_front & unranked), 1)';
        unranked(front) = false; r = r + 1;
    end
end

function cd = crowding_distance_vec(g1_front, g2_front, L)
    cd = zeros(L, 1);
    [~, idx1] = sort(g1_front); cd(idx1(1)) = inf; cd(idx1(end)) = inf;
    r1 = max(g1_front) - min(g1_front) + 1e-9;
    cd(idx1(2:end-1)) = cd(idx1(2:end-1)) + (g1_front(idx1(3:end)) - g1_front(idx1(1:end-2))) / r1;
    [~, idx2] = sort(g2_front); cd(idx2(1)) = inf; cd(idx2(end)) = inf;
    r2 = max(g2_front) - min(g2_front) + 1e-9;
    cd(idx2(2:end-1)) = cd(idx2(2:end-1)) + (g2_front(idx2(3:end)) - g2_front(idx2(1:end-2))) / r2;
end

function perm = greedy_init_fast(dist_mat, costs, N)
    selected = false(N, 1); perm = zeros(1, N, 'int32');
    avg_dist_all = double(sum(dist_mat, 2)) / (N - 1);
    [~, first]   = max(avg_dist_all ./ costs);
    perm(1) = first; selected(first) = true;
    sum_dist = double(dist_mat(:, first));  

    for step = 2:N
        unsel   = find(~selected);
        scores  = sum_dist(unsel) ./ costs(unsel);
        [~, bi] = max(scores); chosen  = unsel(bi);
        perm(step) = chosen; selected(chosen) = true;
        sum_dist = sum_dist + double(dist_mat(:, chosen));
    end
end

function [child1, child2] = pmx_crossover(p1, p2, N)
    cut1 = randi([1, N-1]); cut2 = randi([cut1+1, N]);
    child1 = p1; child2 = p2;
    map12 = zeros(1, N); map21 = zeros(1, N);
    for i = cut1:cut2
        map12(p1(i)) = p2(i); map21(p2(i)) = p1(i);
        child1(i) = p2(i); child2(i) = p1(i);
    end
    for i = [1:cut1-1, cut2+1:N]
        v = p1(i); while map21(v), v = map21(v); end; child1(i) = v;
        v = p2(i); while map12(v), v = map12(v); end; child2(i) = v;
    end
end

function ind = mutate(ind, p_m, N)
    for i = 1:N
        if rand() < p_m
            op = randi(3);
            if op == 1                          
                ab = randperm(N, 2); ind([ab(1) ab(2)]) = ind([ab(2) ab(1)]);
            elseif op == 2                      
                a = randi(N-1); b = randi([a+1, N]); ind(a:b) = ind(b:-1:a);
            else                                
                ab = randperm(N, 2); v = ind(ab(1)); ind(ab(1)) = [];
                pos = min(max(ab(2) - (ab(2) > ab(1)), 1), length(ind)+1);
                ind = [ind(1:pos-1), v, ind(pos:end)];
            end
        end
    end
end

function sel = roulette_select(fitness)
    f = fitness - min(fitness) + 1e-6; c = cumsum(f / sum(f));
    sel = find(c >= rand(), 1); if isempty(sel), sel = 1; end
end

function sel = binary_tournament(ranks, crowding)
    c1 = randi(numel(ranks)); c2 = randi(numel(ranks));
    if ranks(c1) < ranks(c2), sel = c1;
    elseif ranks(c1) > ranks(c2), sel = c2;
    elseif crowding(c1) >= crowding(c2), sel = c1;
    else, sel = c2; end
end