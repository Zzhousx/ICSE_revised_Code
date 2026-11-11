function table_I(provider)
if nargin<1,provider=@load_rank;end
methods={'VoD-Sampler','SO-SDC','MO-SDC','STRaP-IR','STRaP-OS','BehAVExplor','Random'};
rows=cell(0,9);values=zeros(3,numel(methods),3);
for d=1:3
 [~,y,c,~,~,~,root]=paper_data(d);cats=unique(c(y==1 & c>0));
 for j=1:numel(methods)
  n=30;if j==1,n=1;end
  fd=zeros(n,1);ccr=zeros(n,1);cdw=zeros(n,1);
  for s=0:n-1
   r=provider(root,d,methods{j},s);
   fd(s+1)=compute_FaultCurve(r,y,5)/100;
   ccr(s+1)=compute_CCR(r,y,c,.05,cats);
   cdw(s+1)=compute_CDWAPFD(r,y,c);
  end
  values(d,j,:)=reshape([mean(fd),mean(ccr),mean(cdw)],1,1,[]);
  rows(end+1,:)={sprintf('Dataset_%d',d),methods{j},mean(fd),mean(ccr),mean(cdw),round(mean(fd)*100,1),round(mean(ccr)*100,1),NaN,NaN}; %#ok<AGROW>
 end
end
avg=squeeze(mean(values,1));
printed=squeeze(round(mean(round(values(:,:,1:2)*100,1),1),1));
for j=1:numel(methods)
 gainFD=NaN;gainCCR=NaN;
 if j>1,gainFD=printed(1,1)/printed(j,1)-1;gainCCR=printed(1,2)/printed(j,2)-1;end
 rows(end+1,:)={'Average',methods{j},avg(j,1),avg(j,2),avg(j,3),printed(j,1),printed(j,2),gainFD,gainCCR}; %#ok<AGROW>
end
t=cell2table(rows,'VariableNames',{'Dataset','Method','FD_5','CCR_5','CDW_APFD','DisplayedFDPercent','DisplayedCCRPercent','GainFD','GainCCR'});
writetable(t,fullfile(root,'outputs','Table_I.csv'));disp(t);
end
