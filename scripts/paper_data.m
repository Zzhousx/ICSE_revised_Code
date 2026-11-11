function [data, y, c, alpha, window, step, root] = paper_data(dataset)
root = fileparts(fileparts(mfilename('fullpath')));
addpath(root, fullfile(root,'function'), fullfile(root,'experiment'));
if ~isfolder(fullfile(root,'outputs')),mkdir(fullfile(root,'outputs'));end
assert(ismember(dataset,1:3));
data = load_dataset(fullfile(root,'data',sprintf('Dataset_%d.mat',dataset)));
y = [data.Label]'; c = [data.ClassID]';
alphas = [.5 .5 .1]; windows = [5 5 6];
alpha = alphas(dataset); window = windows(dataset); step = 2;
end
