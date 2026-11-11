function gs = compute_GeomSpread(selectedIdx, q_scores, tangentVectors)
    K  = length(selectedIdx);
    Dv = size(tangentVectors, 2);
    B  = zeros(K, Dv);

    for ki = 1:K
        idx       = selectedIdx(ki);
        v         = tangentVectors(idx,:);         
        B(ki,:)   = q_scores(idx) * v;
    end

    G       = B * B' + 1e-10 * eye(K);
    eigvals = real(eig(G));
    eigvals = eigvals(eigvals > 1e-12);
    if isempty(eigvals), gs = -Inf; return; end
    gs = sum(log(eigvals));
end