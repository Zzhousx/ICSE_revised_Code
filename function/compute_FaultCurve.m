function curve = compute_FaultCurve(rankedIndices, groundTruth, budgetPcts)
    N           = length(rankedIndices);
    totalFaults = sum(groundTruth);
    curve       = zeros(length(budgetPcts), 1);
    if totalFaults == 0, return; end

    cumF = cumsum(groundTruth(rankedIndices));  
    for bi = 1:length(budgetPcts)
        K         = min(max(1, round(budgetPcts(bi)/100*N)), N);
        curve(bi) = cumF(K) / totalFaults * 100;
    end
end