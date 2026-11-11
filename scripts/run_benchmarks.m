function run_benchmarks(dataset)
[data,~,~,~,~,~,root] = paper_data(dataset); N=numel(data); rng('default');
out=fullfile(root,'outputs','rankings');if ~isfolder(out),mkdir(out);end
for seed=0:29
 methods={'Random','SO-SDC','MO-SDC','BehAVExplor','STRaP-IR'};
 for j=1:numel(methods)
  switch methods{j}
   case 'Random', rank=randperm(N)';
   case 'SO-SDC', rank=baseline_SDC_GA(data,'SO');
   case 'MO-SDC', rank=baseline_SDC_GA(data,'MO');
   case 'BehAVExplor', rank=baseline_BehAVExplor(data);
   case 'STRaP-IR', rank=baseline_SBT(data);
  end
  rank=rank(:);assert(isequal(sort(rank),(1:N)'));
  save(fullfile(out,sprintf('Dataset_%d_%s_seed%02d.mat',dataset,methods{j},seed)),'rank','seed','dataset');
 end
 fprintf('Dataset_%d input order %d complete: SO-SDC, MO-SDC, STRaP-IR, BehAVExplor, Random.\n',dataset,seed);
end
end
