clc; 
clear; 
close all;
rng('default');  

addpath(".\function");
addpath(".\feature extraction\");
addpath(".\experiment\");

            path1 = '.\feature extraction\DeepScenario_Greedy_Features_with_Labels_new3.mat';
            path2 = '.\feature extraction\DeepScenario_random_Features_with_Labels_new3.mat';
            path3 = '.\feature extraction\DeepScenario_RL_Features_with_Labels_new3.mat';
            results1 = main_evaluation(path1, 0.5, 5, 2);
            results2 = main_evaluation(path2, 0.5, 5, 2);
            results3 = main_evaluation(path3, 0.1, 6, 2);
