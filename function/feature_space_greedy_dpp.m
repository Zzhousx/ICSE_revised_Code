function Y_out = feature_space_greedy_dpp(B, max_k, original_scores)
% FEATURE_SPACE_GREEDY_DPP 
% Phase 1: DPP extracts full-rank bases of the space (Anchors)
% Phase 2: Strict bucket polling based on basis projection (Round-Robin with Skipping)

    [N, D_v] = size(B);
    B_orig = B; 
    
    k_dpp = min([max_k, N, D_v]); 
    Y = zeros(k_dpp, 1);
    marginal_gains = sum(B.^2, 2); 
    % Phase 1
    for i = 1:k_dpp
        [max_val, best_idx] = max(marginal_gains);
        if max_val < 1e-10
            Y = Y(1:i-1);
            break;
        end
        
        Y(i) = best_idx;
        e = B(best_idx, :) / sqrt(max_val);
        
        projections = B * e'; 
        B = B - projections * e; 
        
        marginal_gains = marginal_gains - projections.^2;
        marginal_gains(marginal_gains < 0) = 0; 
        marginal_gains(Y(1:i)) = 0; 
    end
    
    Y_out = Y(Y > 0);
    num_anchors = length(Y_out);

    % Phase 2
    if num_anchors < max_k && num_anchors < N

        rem_idx = setdiff((1:N)', Y_out);
        rem_scores = original_scores(rem_idx);
        
        sim_matrix = B_orig(rem_idx, :) * B_orig(Y_out, :)';
        [~, bucket_assignment] = max(sim_matrix, [], 2);

        buckets = cell(num_anchors, 1);
        for b = 1:num_anchors
            items_in_b = find(bucket_assignment == b);
            [~, sort_idx] = sort(rem_scores(items_in_b), 'descend');
            buckets{b} = rem_idx(items_in_b(sort_idx));
        end
        
        anchor_scores = original_scores(Y_out);
        [~, bucket_poll_order] = sort(anchor_scores, 'descend');
        
        Y_tail = zeros(length(rem_idx), 1);
        pointers = ones(num_anchors, 1);
        filled = 0;
        
        while filled < length(rem_idx)
            advanced = false;
            for i = 1:num_anchors
                b = bucket_poll_order(i); 
                if pointers(b) <= length(buckets{b})
                    filled = filled + 1;
                    Y_tail(filled) = buckets{b}(pointers(b));
                    pointers(b) = pointers(b) + 1;
                    advanced = true;
                end
            end
            if ~advanced, break; end 
        end
     
        needed = max_k - num_anchors;
        Y_out = [Y_out; Y_tail(1:min(needed, length(Y_tail)))];
    end
end