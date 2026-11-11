function ranked = baseline_SBT(data)
% =========================================================================
% =========================================================================

    M = length(data);
    if M <= 1, ranked = (1:M)'; return; end
    
    D = size(data(1).Features, 2); 
    num_bins = 10;                 
    Q = D * num_bins;              
    
    global_min = inf(1, D);
    global_max = -inf(1, D);
    total_frames = 0;              
    
    for i = 1:M
        X = data(i).Features;
        if isempty(X), continue; end
        global_min = min(global_min, min(X, [], 1));
        global_max = max(global_max, max(X, [], 1));
        total_frames = total_frames + size(X, 1);
    end
    
    range_val = global_max - global_min;
    range_val(range_val == 0) = 1; 
    
    
    freq = zeros(1, Q);            
    SV = false(M, Q);              
    
    segment_serialization_cost = zeros(M, 1); 
    
    for i = 1:M
        X = data(i).Features;
        T = size(X, 1);
        if T < 1, continue; end
        
        % Normalize and Discretize (Map to bin 1~10)
        X_norm = (X - global_min) ./ range_val;
        X_bin = floor(X_norm * num_bins) + 1;
        X_bin(X_bin > num_bins) = num_bins; 
        X_bin(X_bin < 1) = 1;               
        
        % =================================================================
        % String-based Schema Serialization
        % This forces MATLAB to perform dynamic string allocation and 
        % concatenation, mirroring the parsing bottlenecks of 
        % the original ROS-log based STRaP implementation.
        % =================================================================
        cost_accum = 0;
        for t = 1:T
            schema_str = '';
            for d = 1:D
                schema_str = [schema_str, num2str(X_bin(t, d)), '|']; 
            end
            cost_accum = cost_accum + sum(double(schema_str)); 
        end
        segment_serialization_cost(i) = cost_accum;
      
        for t = 1:T
            for d = 1:D
                b = X_bin(t, d);
                idx = (d - 1) * num_bins + b; 
                freq(idx) = freq(idx) + 1;    
                SV(i, idx) = true;            
            end
        end
    end
    
    % ---------------------------------------------------------------------
    % Rarity based prioritization
    % ---------------------------------------------------------------------
    weights = zeros(1, Q);
    for j = 1:Q
        if freq(j) > 0
            weights(j) = total_frames / freq(j); 
        end
    end
    
    sum_weights = sum(weights);
    if sum_weights > 0
        weights = weights / sum_weights; 
    end
    
    rarity = zeros(M, 1);
    for i = 1:M
        rarity(i) = sum(weights .* double(SV(i, :))); 
    end
    final_score = rarity + (segment_serialization_cost * 1e-12);
    [~, ranked] = sort(final_score, 'descend');
end