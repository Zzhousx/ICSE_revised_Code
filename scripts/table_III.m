function table_III
variants={'w/o Quality','w/o Riemannian','w/o Diversity','Ours'};rows=cell(0,5);values=zeros(3,4,3);
for d=1:3
 [data,y,c,alpha,window,step,root]=paper_data(d);cats=unique(c(y==1 & c>0));
 for j=1:numel(variants)
  r=VoD_Sampler_ablation(data,alpha,window,step,variants{j});
  cdw=compute_CDWAPFD(r,y,c);fd=compute_FaultCurve(r,y,5)/100;ccr=compute_CCR(r,y,c,.05,cats);
  name=variants{j};if strcmp(name,'Ours'),name='VoD-Sampler';end
  values(d,j,:)=reshape([cdw,fd,ccr],1,1,[]);rows(end+1,:)={sprintf('Dataset_%d',d),name,cdw,fd,ccr}; %#ok<AGROW>
 end
end
avg=squeeze(mean(values,1));
for j=1:numel(variants)
 name=variants{j};if strcmp(name,'Ours'),name='VoD-Sampler';end
 rows(end+1,:)={'Average',name,avg(j,1),avg(j,2),avg(j,3)}; %#ok<AGROW>
end
t=cell2table(rows,'VariableNames',{'Dataset','Variant','CDW_APFD','FD_5','CCR_5'});
writetable(t,fullfile(root,'outputs','Table_III.csv'));disp(t);
end
