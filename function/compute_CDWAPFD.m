function apfdc = compute_CDWAPFD(rankedIndices, groundTruth, classIDs)
    rankedIndices = rankedIndices(:);
    groundTruth   = groundTruth(:);
    classIDs      = classIDs(:);
    
    N = length(rankedIndices);
    ordered_GT   = groundTruth(rankedIndices);
    ordered_Cats = classIDs(rankedIndices);
    weights = zeros(N, 1);
    maxCatID = max(classIDs); 
    if isempty(maxCatID) || maxCatID <= 0
        apfdc = 0; 
        return; 
    end
    cat_seen_count = zeros(maxCatID, 1);
    
    for i = 1:N
        if ordered_GT(i) == 1 && ordered_Cats(i) > 0
            c = ordered_Cats(i);
            cat_seen_count(c) = cat_seen_count(c) + 1;
            weights(i) = 1.0 / log2(cat_seen_count(c) + 1);
        end
    end
    S = cumsum(weights);   
    W_total = S(end);      
    
    if W_total == 0
        apfdc = 0;
        return;
    end

    S_norm = S / W_total;  
    
    apfdc = (sum(S_norm) - 0.5) / N;
end


