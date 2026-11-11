function run_method(dataset, method, seed)
if nargin < 3, seed = 0; end
[data,~,~,alpha,window,step,root] = paper_data(dataset);
assert(isscalar(seed) && seed>=0 && seed<=29 && seed==floor(seed));
rng(seed,'twister'); N = numel(data); V=[]; q=[];
switch method
 case 'VoD-Sampler'
  [rank,V,q] = VoD_Sampler_core(data,alpha,window,step);
 case 'SO-SDC'
  rank = baseline_SDC_GA(data,'SO');
 case 'MO-SDC'
  rank = baseline_SDC_GA(data,'MO');
 case 'STRaP-IR'
  rank = baseline_SBT(data);
 case 'BehAVExplor'
  rank = baseline_BehAVExplor(data);
 case 'Random'
  rank = randperm(N)';
 otherwise
  error('Unknown method. Use the STRaP-OS Python entry for STRaP-OS.');
end
rank = rank(:); assert(isequal(sort(rank),(1:N)'));
out = fullfile(root,'outputs','rankings'); if ~isfolder(out),mkdir(out);end
save(fullfile(out,sprintf('Dataset_%d_%s_seed%02d.mat',dataset,method,seed)),'rank','V','q','seed','dataset');
fprintf('Dataset_%d %s input order %d: ranking saved in outputs/rankings/.\n',dataset,method,seed);
end
