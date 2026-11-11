function table_II(provider)
if nargin<1,provider=@load_rank;end
methods={'VoD-Sampler','SO-SDC','MO-SDC','STRaP-IR','STRaP-OS','BehAVExplor','Random'};
budgets=[5 10 15 20];rows=cell(0,4);values=zeros(3,numel(methods),numel(budgets));
for d=1:3
 [data,~,~,alpha,window,step,root]=paper_data(d);N=numel(data);
 p=fullfile(root,'outputs','rankings',sprintf('Dataset_%d_VoD-Sampler_seed00.mat',d));
 if ~isfile(p),run_method(d,'VoD-Sampler',0);end
 v=load(p,'V','q');assert(~isempty(v.V) && ~isempty(v.q));
 for bi=1:numel(budgets)
  b=budgets(bi);
  K=round(N*b/100);rand_gs=zeros(30,1);
  for seed=0:29,r=provider(root,d,'Random',seed);rand_gs(seed+1)=compute_GeomSpread(r(1:K),v.q,v.V);end
  reference=mean(rand_gs);
  for j=1:numel(methods)
   n=1;if strcmp(methods{j},'STRaP-OS') || strcmp(methods{j},'Random'),n=30;end
   gs=zeros(n,1);
   for seed=0:n-1,r=provider(root,d,methods{j},seed);gs(seed+1)=compute_GeomSpread(r(1:K),v.q,v.V);end
   values(d,j,bi)=mean(gs)-reference;
   rows(end+1,:)={sprintf('Dataset_%d',d),methods{j},b,values(d,j,bi)}; %#ok<AGROW>
  end
 end
end
avg=squeeze(mean(values,1));
for bi=1:numel(budgets)
 for j=1:numel(methods),rows(end+1,:)={'Average',methods{j},budgets(bi),avg(j,bi)};end %#ok<AGROW>
 best=max(avg(2:end,bi));rows(end+1,:)={'Average','Ratio vs Best Baseline',budgets(bi),avg(1,bi)/best}; %#ok<AGROW>
end
t=cell2table(rows,'VariableNames',{'Dataset','Method','BudgetPercent','DeltaGS'});
writetable(t,fullfile(root,'outputs','Table_II.csv'));disp(t);
end
