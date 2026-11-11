function section_V_A(provider)
if nargin<1,provider=@load_rank;end
methods={'VoD-Sampler','SO-SDC','MO-SDC','STRaP-IR','STRaP-OS','BehAVExplor','Random'};rows=cell(0,6);
for d=1:3
 [~,y,c,~,~,~,root]=paper_data(d);base=c;base(base>0)=mod(base(base>0)-1,17)+1;
 cats=unique(base(y==1 & base>0));v=provider(root,d,'VoD-Sampler',0);
 vod=compute_CDWAPFD(v,y,c);
 for j=1:numel(methods)
  n=30;if j==1,n=1;end
  coverage=zeros(n,1);scores=zeros(n,1);
  for s=0:n-1
   r=provider(root,d,methods{j},s);
   coverage(s+1)=compute_CCR(r,y,base,.05,cats);scores(s+1)=compute_CDWAPFD(r,y,c);
  end
  if j==1,a=NaN;p=NaN;else,[a,p]=compute_StatTest(vod,scores);end
  rows(end+1,:)={sprintf('Dataset_%d',d),methods{j},mean(coverage),mean(scores),a,p}; %#ok<AGROW>
 end
end
t=cell2table(rows,'VariableNames',{'Dataset','Method','BaseCoverage5','CDW_APFD','A12','RankSumP'});
writetable(t,fullfile(root,'outputs','Section_V_A.csv'));disp(t);
end
