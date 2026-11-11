clc; 
clear; 
close all;
rng('default');  

addpath(".\function");
addpath(".\feature extraction\");
addpath(".\experiment\");

datasets = {
    '.\feature extraction\DeepScenario_Greedy_Features_with_Labels_new3.mat', 0.5, 5, 2;
    '.\feature extraction\DeepScenario_random_Features_with_Labels_new3.mat', 0.5, 5, 2;
    '.\feature extraction\DeepScenario_RL_Features_with_Labels_new3.mat', 0.1, 6, 2
};

for ds_idx = 1:size(datasets, 1)
    datasetPath = datasets{ds_idx, 1};
    alpha       = datasets{ds_idx, 2};
    window_size = datasets{ds_idx, 3};
    step_size   = datasets{ds_idx, 4};
    
    fprintf('\n======================================================\n');
    fprintf('Starting Ablation Study for Dataset %d\n', ds_idx);
    fprintf('======================================================\n');
    
    results = run_ablation(datasetPath, alpha, window_size, step_size);
end


function results = run_ablation(datasetPath, alpha, window_size, step_size)

    fprintf('[1/4] Load the dataset: %s\n', datasetPath);
    data = load_dataset(datasetPath);
    N           = length(data);
    groundTruth = [data.Label]';  
    classIDs    = [data.ClassID]';  

    populated_cats = unique(classIDs(groundTruth == 1 & classIDs > 0));
    fprintf('      Total scenes: %d | Failures: %d | Categories: %d\n', ...
            N, sum(groundTruth), length(populated_cats));

    variants = {'Ours', 'w/o Diversity', 'w/o Riemannian', 'w/o Quality'};
    num_vars = length(variants);

    budgetPcts = 0:0.5:100;
    kPcts      = [5, 10, 15, 20, 25];
    
    cdwapfd_results = zeros(num_vars, 1);
    curve_results   = zeros(length(budgetPcts), num_vars);
    ccr_results     = zeros(length(kPcts), num_vars);
    
    fprintf('[2/4] Running ablation variants...\n');
    
    for v = 1:num_vars
        current_var = variants{v};
        fprintf('      > Running %-15s ... ', current_var);
        
        tic;
        [rankedIndices, ~, ~] = VoD_Sampler_ablation(data, alpha, window_size, step_size, current_var);
        t_elapsed = toc;
        fprintf('Done (%.2f s)\n', t_elapsed);
        
        cdwapfd_results(v) = compute_CDWAPFD(rankedIndices, groundTruth, classIDs);
        curve_results(:, v) = compute_FaultCurve(rankedIndices, groundTruth, budgetPcts);
        ccr_results(:, v) = compute_CCR(rankedIndices, groundTruth, classIDs, kPcts, length(populated_cats));
    end

    fprintf('[3/4] Extracting metrics at 5%% execution budget...\n');
    idx5 = find(abs(budgetPcts - 5) < 1e-6, 1);
    idx_k5 = find(kPcts == 5, 1);
    
    fd_at_5  = curve_results(idx5, :);
    ccr_at_5 = ccr_results(idx_k5, :);

    fprintf('\n=== Ablation Results Summary (5%% Budget) ===\n');
    fprintf('%-18s | %-10s | %-10s | %-10s\n', 'Variant', 'CDW-APFD', 'FD@5%', 'CCR@5%');
    fprintf('-------------------|------------|------------|------------\n');
    for v = 1:num_vars
        if strcmp(variants{v}, 'Ours')
            fprintf('%-18s | %-10.3f | %-9.2f%% | %-9.2f%%\n', ...
                ['**' variants{v} '**'], cdwapfd_results(v), fd_at_5(v), ccr_at_5(v));
        else
            fprintf('%-18s | %-10.3f | %-9.2f%% | %-9.2f%%\n', ...
                variants{v}, cdwapfd_results(v), fd_at_5(v), ccr_at_5(v));
        end
    end
    fprintf('==============================================\n\n');

    fprintf('[4/4] Plotting Ablation Curves...\n');
    curves_cell = num2cell(curve_results, 1); 
    apfds_array = cdwapfd_results';         

    plot_FaultCurves(budgetPcts, curves_cell, apfds_array, variants);

    results.variants = variants;
    results.cdwapfd  = cdwapfd_results;
    results.curves   = curve_results;
    results.ccr      = ccr_results;
end