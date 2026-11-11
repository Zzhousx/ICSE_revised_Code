function results = main_evaluation(datasetPath, alpha, window_size, step_size, nRuns)
% =========================================================================
% run_evaluation  —  VoD-Sampler complete assessment entry
%
%   HOW TO CALL：
%   results = run_evaluation(datasetPath, alpha, window_size, step_size)
%   results = run_evaluation(datasetPath, alpha, window_size, step_size, nRuns)
% =========================================================================

    if nargin < 5, nRuns = 30; end
    rng('default');

    fprintf('[1/7] Load the dataset: %s\n', datasetPath);
    data = load_dataset(datasetPath);
    N          = length(data);
    groundTruth = [data.Label]';  
    classIDs    = [data.ClassID]';  

    populated_cats = unique(classIDs(groundTruth == 1 & classIDs > 0));
    fprintf('      Total number of scenes N=%d; Failure scenario=%d; The actual number of categories that occurred=%d\n', ...
            N, sum(groundTruth),length(populated_cats));


    fprintf('[2/7] Run the proposed VoD-Sampler (alpha=%.1f, tau=%d, s=%d)...\n', alpha, window_size, step_size);
    tic
    [rankedIndices_PIRDPP, tangentVectors, q_scores] = VoD_Sampler_core(data, alpha, window_size, step_size);
    toc

    % 3. Generate all baseline rankings
    [~, ds_name, ~] = fileparts(datasetPath);
    cache_baselines_test = sprintf('Cache_Baselines_test_%s_runs%d.mat', ds_name, nRuns);

    need_recompute = true; 

    if exist(cache_baselines_test, 'file')
        fprintf('[3/7] A baseline cache file has been detected: %s. Attempting to load...\n', cache_baselines_test);
        loaded_data = load(cache_baselines_test);
        
        if isfield(loaded_data, 'rankedIndices_Random') && ...
           size(loaded_data.rankedIndices_Random, 1) == N && ...
           size(loaded_data.rankedIndices_Random, 2) == nRuns

            rankedIndices_Random = loaded_data.rankedIndices_Random;
            rankedIndices_SO     = loaded_data.rankedIndices_SO;
            rankedIndices_MO     = loaded_data.rankedIndices_MO;
            rankedIndices_behave = loaded_data.rankedIndices_behave;
            rankedIndices_dofuzz = loaded_data.rankedIndices_dofuzz;
            
            fprintf('      > [Success] Cache dimension verification passed (N=%d, runs=%d), loading historical results.\n', N, nRuns);
            need_recompute = false;
        else
            fprintf('      > [warning] Cache data dimensions do not match. \n');
            fprintf('      > Prepare to re-calculate %d times...\n', nRuns);
            delete(cache_baselines_test); 
        end
    end

    if need_recompute
        fprintf('[3/7] Start to execute the heuristic baseline (generating a %d-column matrix)...\n', nRuns);
        
        rankedIndices_Random  = zeros(N, nRuns);
        rankedIndices_SO      = zeros(N, nRuns);
        rankedIndices_MO      = zeros(N, nRuns);
        rankedIndices_behave  = zeros(N, nRuns);
        rankedIndices_dofuzz  = zeros(N, nRuns);
        
        for r = 1:nRuns
            fprintf('      > Progress: %d / %d runs completed\n', r, nRuns);
            
            % 1. Random
            rankedIndices_Random(:,r) = randperm(N)';

            % 2. SO-SDC
            fprintf('        - SO-SDC...\n');
            tic
            temp_SO = baseline_SDC_GA(data, 'SO'); 
            rankedIndices_SO(:,r) = temp_SO(:);
            toc

            % 3. MO-SDC
            fprintf('        - MO-SDC...\n');
            tic
            temp_MO = baseline_SDC_GA(data, 'MO');
            rankedIndices_MO(:,r) = temp_MO(:);
            toc

            % 4. Behavior-Novelty
            fprintf('        - BehAVExplor...\n');
            tic
            temp_behave = baseline_BehAVExplor(data);
            toc
            rankedIndices_behave(:,r) = temp_behave(:);
            
            % 5. STRaP
            fprintf('        - STRaP...\n');
            tic
            temp_dofuzz = baseline_SBT(data);
            toc
            rankedIndices_dofuzz(:,r) = temp_dofuzz(:);
        end
        save(cache_baselines_test, 'rankedIndices_Random', 'rankedIndices_SO', 'rankedIndices_MO', ...
                              'rankedIndices_behave', 'rankedIndices_dofuzz');
        fprintf('      > The baseline has been generated. The %d column matrix has been saved!\n', nRuns);
    end


    % 4. Calculate CDW-APFD
    fprintf('[4/7] Calculate CDW-APFD ...\n');
    cdwapfd_PIRDPP = compute_CDWAPFD(rankedIndices_PIRDPP, groundTruth, classIDs);
    cdwapfd_Random = zeros(nRuns,1);
    cdwapfd_SO     = zeros(nRuns,1);
    cdwapfd_MO     = zeros(nRuns,1);
    cdwapfd_dofuzz = zeros(nRuns,1);
    cdwapfd_behave = zeros(nRuns,1);

    for r = 1:nRuns
        cdwapfd_Random(r) = compute_CDWAPFD(rankedIndices_Random(:,r), groundTruth, classIDs);
        cdwapfd_SO(r)     = compute_CDWAPFD(rankedIndices_SO(:,r), groundTruth, classIDs);
        cdwapfd_MO(r)     = compute_CDWAPFD(rankedIndices_MO(:,r), groundTruth, classIDs);
        cdwapfd_dofuzz(r) = compute_CDWAPFD(rankedIndices_dofuzz(:,r), groundTruth, classIDs);
        cdwapfd_behave(r) = compute_CDWAPFD(rankedIndices_behave(:,r), groundTruth, classIDs);
    end

    % 5. Fault detection curve + CCR@K + Geometric Spread
    fprintf('[5/7] Computational Curve / CCR / Geometric Spread...\n');
    budgetPcts = 0:0.5:100;
    kPcts      = [5, 10, 15, 20, 25];
    kBudgets   = round(kPcts / 100 * N);

    curve_PIRDPP = compute_FaultCurve(rankedIndices_PIRDPP, groundTruth, budgetPcts);
    curve_Random = zeros(length(budgetPcts), nRuns);
    curve_SO     = zeros(length(budgetPcts), nRuns);
    curve_MO     = zeros(length(budgetPcts), nRuns);
    curve_dofuzz = zeros(length(budgetPcts), nRuns);
    curve_behave = zeros(length(budgetPcts), nRuns);

    for r = 1:nRuns
        curve_SO(:,r)     = compute_FaultCurve(rankedIndices_SO(:,r), groundTruth, budgetPcts);
        curve_MO(:,r)     = compute_FaultCurve(rankedIndices_MO(:,r), groundTruth, budgetPcts);
        curve_dofuzz(:,r) = compute_FaultCurve(rankedIndices_dofuzz(:,r), groundTruth, budgetPcts);
        curve_behave(:,r) = compute_FaultCurve(rankedIndices_behave(:,r), groundTruth, budgetPcts);
        curve_Random(:,r) = compute_FaultCurve(rankedIndices_Random(:,r), groundTruth, budgetPcts);
    end

    idx30 = find(abs(budgetPcts - 5) < 1e-6, 1);
    pct30_PIRDPP = curve_PIRDPP(idx30);
    pct30_SO     = mean(curve_SO(idx30,:));
    pct30_MO     = mean(curve_MO(idx30,:));
    pct30_dofuzz = mean(curve_dofuzz(idx30,:));
    pct30_behave = mean(curve_behave(idx30,:));
    pct30_Random = mean(curve_Random(idx30,:));

    numCats = unique(classIDs(groundTruth == 1 & classIDs > 0));
    ccr_PIRDPP = compute_CCR(rankedIndices_PIRDPP, groundTruth, classIDs, kPcts, numCats);
    ccr_SO     = zeros(length(kPcts), nRuns);
    ccr_MO     = zeros(length(kPcts), nRuns);
    ccr_dofuzz = zeros(length(kPcts), nRuns);
    ccr_behave = zeros(length(kPcts), nRuns);
    ccr_Random = zeros(length(kPcts), nRuns);

    for r = 1:nRuns
        ccr_Random(:,r) = compute_CCR(rankedIndices_Random(:,r), groundTruth, classIDs, kPcts, numCats);
        ccr_SO(:,r)     = compute_CCR(rankedIndices_SO(:,r), groundTruth, classIDs, kPcts, numCats);
        ccr_MO(:,r)     = compute_CCR(rankedIndices_MO(:,r), groundTruth, classIDs, kPcts, numCats);
        ccr_dofuzz(:,r) = compute_CCR(rankedIndices_dofuzz(:,r), groundTruth, classIDs, kPcts, numCats);
        ccr_behave(:,r) = compute_CCR(rankedIndices_behave(:,r), groundTruth, classIDs, kPcts, numCats);
    end

    gs_PIRDPP = zeros(length(kBudgets),1);
    gs_SO     = zeros(length(kBudgets),1);
    gs_MO     = zeros(length(kBudgets),1);
    gs_dofuzz = zeros(length(kBudgets),1);
    gs_behave = zeros(length(kBudgets),1);
    gs_Random = zeros(length(kBudgets), nRuns);
    
    for ki = 1:length(kBudgets)
        K = kBudgets(ki);
        gs_PIRDPP(ki) = compute_GeomSpread(rankedIndices_PIRDPP(1:K), q_scores, tangentVectors);
        gs_SO(ki)     = compute_GeomSpread(rankedIndices_SO(1:K), q_scores, tangentVectors);
        gs_MO(ki)     = compute_GeomSpread(rankedIndices_MO(1:K), q_scores, tangentVectors);
        gs_dofuzz(ki) = compute_GeomSpread(rankedIndices_dofuzz(1:K), q_scores, tangentVectors);
        gs_behave(ki) = compute_GeomSpread(rankedIndices_behave(1:K), q_scores, tangentVectors);

        for r = 1:nRuns
            gs_Random(ki,r) = compute_GeomSpread(rankedIndices_Random(1:K,r), q_scores, tangentVectors);
        end
    end
    fprintf('[6/7] Conduct statistical testing...\n');
    
    [a12_R, p_R] = compute_StatTest(cdwapfd_PIRDPP, cdwapfd_Random);
    [a12_SO, p_SO] = compute_StatTest(cdwapfd_PIRDPP, cdwapfd_SO);
    [a12_MO, p_MO] = compute_StatTest(cdwapfd_PIRDPP, cdwapfd_MO);
    [a12_behave, p_behave] = compute_StatTest(cdwapfd_PIRDPP, cdwapfd_behave);
    [a12_cluster, p_cluster] = compute_StatTest(cdwapfd_PIRDPP, cdwapfd_dofuzz);


    % 7. Print reports 
    fprintf('[7/7] Output result...\n');
    print_report(cdwapfd_PIRDPP, cdwapfd_Random, cdwapfd_SO, cdwapfd_MO, cdwapfd_dofuzz, cdwapfd_behave, ...
                 a12_R, p_R, a12_SO, p_SO, a12_MO, p_MO, a12_cluster, p_cluster, a12_behave, p_behave, ...
                 kPcts, ccr_PIRDPP, ccr_Random, ccr_SO, ccr_MO, ccr_dofuzz, ccr_behave,  ...
                 kBudgets, gs_PIRDPP, gs_Random, gs_SO, gs_MO, gs_dofuzz, gs_behave,  ...
                 pct30_PIRDPP, mean(curve_Random(idx30,:)), pct30_SO, pct30_MO, pct30_dofuzz, pct30_behave);

    apfds1 = [cdwapfd_PIRDPP,  mean(cdwapfd_MO),  mean(cdwapfd_SO), mean(cdwapfd_dofuzz),   mean(cdwapfd_behave), mean(cdwapfd_Random)];
    labels1 = {'VoD-Sampler', 'MO-SDC', 'SO-SDC', 'STRaP', 'BehAVExplor', 'Random'};
    curves_cell1 = {curve_PIRDPP, ...
                   mean(curve_MO, 2), ...
                   mean(curve_SO, 2), ...
                   mean(curve_dofuzz, 2), ...
                   mean(curve_behave, 2), ...
                   mean(curve_Random, 2)};
    plot_FaultCurves(budgetPcts, curves_cell1, apfds1, labels1);

    results.cdwapfd.PIRDPP = cdwapfd_PIRDPP;
    results.cdwapfd.MO     = cdwapfd_MO;
    results.cdwapfd.SO     = cdwapfd_SO;
    results.cdwapfd.behave = cdwapfd_behave;
    results.cdwapfd.dofuzz = cdwapfd_dofuzz;
    results.cdwapfd.Random = mean(cdwapfd_Random);

    fprintf('=== Evaluation completed ===\n');
end
