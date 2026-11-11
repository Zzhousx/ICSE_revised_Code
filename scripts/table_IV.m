function table_IV(datasets,runs,methods)
if nargin<1,datasets=1:3;end
if nargin<2,runs=30;end
if nargin<3,methods={'VoD-Sampler','SO-SDC','MO-SDC','STRaP-IR','BehAVExplor'};end
assert(all(ismember(datasets,1:3)) && runs>=1 && runs<=30 && runs==floor(runs));
rows=cell(0,5);
for d=datasets
 [data,~,~,alpha,window,step,root]=paper_data(d);
 rng('default');
 if any(strcmp(methods,'VoD-Sampler'))
  for seed=0:runs-1
   tic;r=VoD_Sampler_core(data,alpha,window,step);
   elapsed=toc;assert(numel(r)==numel(data));
   rows(end+1,:)={sprintf('Dataset_%d',d),'VoD-Sampler',seed,elapsed,'MATLAB'}; %#ok<AGROW>
  end
 end
 order={'SO-SDC','MO-SDC','BehAVExplor','STRaP-IR'};
 for seed=0:runs-1
  randperm(numel(data));
  for j=1:numel(order)
   if ~any(strcmp(methods,order{j})),continue;end
   tic;
   switch order{j}
    case 'SO-SDC', r=baseline_SDC_GA(data,'SO');
    case 'MO-SDC', r=baseline_SDC_GA(data,'MO');
    case 'STRaP-IR', r=baseline_SBT(data);
    case 'BehAVExplor', r=baseline_BehAVExplor(data);
   end
   elapsed=toc;assert(numel(r)==numel(data));
   rows(end+1,:)={sprintf('Dataset_%d',d),order{j},seed,elapsed,'MATLAB'}; %#ok<AGROW>
  end
  fprintf('Dataset_%d timing run %d complete.\n',d,seed);
 end
end
t=cell2table(rows,'VariableNames',{'Dataset','Method','Seed','Seconds','Language'});
t.Method=categorical(t.Method,{'VoD-Sampler','SO-SDC','MO-SDC','STRaP-IR','BehAVExplor'},'Ordinal',true);
t=sortrows(t,{'Dataset','Method','Seed'});
writetable(t,fullfile(root,'outputs','Table_IV_runs.csv'));disp(groupsummary(t,{'Dataset','Method'},{'median','mean'},'Seconds'));
end
