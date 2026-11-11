function ccr = compute_CCR(rankedIndices, groundTruth, classIDs, kPcts,numCats)
    rankedIndices = rankedIndices(:);
    groundTruth   = groundTruth(:);
    classIDs      = classIDs(:);
    faultMask     = (groundTruth == 1);
    allFaultCats  = classIDs(faultMask);
    allFaultCats  = allFaultCats(allFaultCats > 0); 
    numActualCats = numel(unique(allFaultCats));

    if numActualCats == 0
        ccr = zeros(length(kPcts), 1);
        return;
    end

    N   = length(rankedIndices);
    ccr = zeros(length(kPcts), 1);

    for ki = 1:length(kPcts)
      
        ratio = kPcts(ki);
        if ratio > 1
            ratio = ratio / 100.0; 
        end

        K    = min(max(1, round(ratio * N)), N);
        topK = rankedIndices(1:K);

        topCats = classIDs(topK);             
        isFault = (groundTruth(topK) == 1);   
        
        validCats = topCats(isFault);        
        validCats = validCats(validCats > 0);  

        ccr(ki) = numel(unique(validCats)) / numActualCats;
    end
end
