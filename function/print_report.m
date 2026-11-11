function print_report(cdwapfd_P, cdwapfd_R, cdwapfd_SO, cdwapfd_MO, cdwapfd_dofuzz, cdwapfd_behave,  ...
                       a12_R, p_R, a12_SO, p_SO, a12_MO, p_MO, a12_cluster, p_cluster, a12_behave, p_behave, ...
                       kPcts, ccr_P, ccr_R, ccr_SO, ccr_MO, ccr_dofuzz, ccr_behave,  ...
                       kBudgets, gs_P, gs_R, gs_SO, gs_MO, gs_dofuzz, gs_behave,  ...
                       pct30_P, pct30_R, pct30_SO, pct30_MO, pct30_dofuzz, pct30_behave)

    sep = repmat('=', 1, 75);
    fprintf('\n%s\n', sep);
    fprintf('  CDW-APFD (Category-Diversity Weighted APFD)\n');
    fprintf('%s\n', sep);
    fprintf('  VoD-Sampler : %.4f\n', mean(cdwapfd_P(:)));
    fprintf('  MO-SDC      : %.4f ± %.4f\n', mean(cdwapfd_MO(:)), std(cdwapfd_MO(:)));
    fprintf('  SO-SDC      : %.4f ± %.4f\n', mean(cdwapfd_SO(:)), std(cdwapfd_SO(:)));
    fprintf('  BehAVExplor : %.4f ± %.4f\n', mean(cdwapfd_behave(:)), std(cdwapfd_behave(:)));
    fprintf('  STRaP       : %.4f ± %.4f\n', mean(cdwapfd_dofuzz(:)), std(cdwapfd_dofuzz(:)));
    fprintf('  Random      : %.4f ± %.4f\n', mean(cdwapfd_R(:)), std(cdwapfd_R(:)));

    fprintf('\n%s\n', sep);
    fprintf('  Fault Detection @30%% Execution Budget\n');
    fprintf('%s\n', sep);
    fprintf('  VoD-Sampler : %.1f%%\n', mean(pct30_P(:)));
    fprintf('  MO-SDC      : %.1f%% ± %.1f%%\n', mean(pct30_MO(:)), std(pct30_MO(:)));
    fprintf('  SO-SDC      : %.1f%% ± %.1f%%\n', mean(pct30_SO(:)), std(pct30_SO(:)));
    fprintf('  BehAVExplor : %.1f%% ± %.1f%%\n', mean(pct30_behave(:)), std(pct30_behave(:)));
    fprintf('  STRaP       : %.1f%% ± %.1f%%\n', mean(pct30_dofuzz(:)), std(pct30_dofuzz(:)));
    fprintf('  Random      : %.1f%% ± %.1f%%\n', mean(pct30_R(:)), std(pct30_R(:)));
    

    fprintf('\n%s\n', sep);
    fprintf('  Statistical Tests (Â₁₂ + Wilcoxon rank-sum against RiST-AD)\n');
    fprintf('%s\n', sep);
    fprintf('  vs MO-SDC        : Â₁₂=%.2f  p=%.2e\n', a12_MO, p_MO);
    fprintf('  vs SO-SDC        : Â₁₂=%.2f  p=%.2e\n', a12_SO, p_SO);
    fprintf('  vs BehAVExplor   : Â₁₂=%.2f  p=%.2e\n', a12_behave, p_behave);
    fprintf('  vs STRaP         : Â₁₂=%.2f  p=%.2e\n', a12_cluster, p_cluster);
    fprintf('  vs Random        : Â₁₂=%.2f  p=%.2e\n', a12_R, p_R);

    fprintf('\n%s\n', sep);
    fprintf('  CCR@K (Category Coverage Rate)\n');
    fprintf('%s\n', sep);
    fprintf('  K%%\t VoD-Sampler   MO-SDC    SO-SDC    BehAVExplor   STRaP    Random\n');
    for ki = 1:length(kPcts)
        fprintf(' %3d%%\t  %.3f     %.3f     %.3f     %.3f    %.3f    %.3f\n', ...
                kPcts(ki), ...
                mean(ccr_P(ki,:)), mean(ccr_MO(ki,:)), mean(ccr_SO(ki,:)), ...
                mean(ccr_behave(ki,:)), mean(ccr_dofuzz(ki,:)), ...
                mean(ccr_R(ki,:)));
    end

    fprintf('\n%s\n', sep);
    fprintf('  Relative Geometric Spread (Delta GS over Random Baseline)\n');
    fprintf('%s\n', sep);
    fprintf('  K%%\t VoD-Sampler   MO-SDC   SO-SDC   BehAVExplor  STRaP  Random\n');
    
    mean_gs_R = mean(gs_R, 2); 
    for ki = 1:length(kBudgets)
        delta_P      = mean(gs_P(ki,:))      - mean_gs_R(ki);
        delta_MO     = mean(gs_MO(ki,:))     - mean_gs_R(ki);
        delta_SO     = mean(gs_SO(ki,:))     - mean_gs_R(ki);
        delta_behave = mean(gs_behave(ki,:)) - mean_gs_R(ki);
        delta_dofuzz = mean(gs_dofuzz(ki,:)) - mean_gs_R(ki);

        fprintf(' %3d%%\t %+7.2f  %+7.2f  %+7.2f  %+7.2f  %+7.2f  +0.00\n', ...
                kPcts(ki), delta_P, delta_MO, delta_SO, delta_behave, delta_dofuzz);
    end
end