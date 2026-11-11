function [a12, p] = compute_StatTest(score_method, scores_baseline)
    nRuns = length(scores_baseline);
    wins  = sum(score_method > scores_baseline);
    ties  = sum(score_method == scores_baseline);
    a12   = (wins + 0.5*ties) / nRuns;

    % Wilcoxon
    group1 = repmat(score_method, nRuns, 1);
    [p, ~] = ranksum(group1, scores_baseline(:), 'tail', 'right');
end
