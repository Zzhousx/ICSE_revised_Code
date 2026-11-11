function figure3(provider)
if nargin<1,provider=@load_rank;end
methods={'VoD-Sampler','SO-SDC','MO-SDC','STRaP-IR','STRaP-OS','BehAVExplor','Random'};
plot_order=[1 3 2 4 6 7 5]; desired=[1 3 2 4 7 5 6];
grid=0:.5:100;
for d=1:3
 [~,y,c,~,~,~,root]=paper_data(d); curves=cell(1,7);scores=zeros(1,7);
 for j=1:7
  method=methods{plot_order(j)};n=30;if strcmp(method,'VoD-Sampler'),n=1;end
  f=zeros(numel(grid),n); a=zeros(n,1);
  for seed=0:n-1
   rank=provider(root,d,method,seed);f(:,seed+1)=compute_FaultCurve(rank,y,grid);a(seed+1)=compute_CDWAPFD(rank,y,c);
  end
  curves{j}=mean(f,2);scores(j)=mean(a);
 end
 labels=methods(plot_order);plot_FaultCurves(grid,curves,scores,labels);fig=gcf;
 axeslist=findall(fig,'Type','axes'); positions=arrayfun(@(x)x.Position,axeslist,'UniformOutput',false);
 main=axeslist(find(cellfun(@(x)max(abs(x-[.12 .14 .83 .80]))<1e-6,positions),1));
 xlabel(main,'Inspection Budget (%)','FontSize',18);
 lines=findall(main,'Type','line');ordered=gobjects(1,7);
 for j=1:7
  matches=arrayfun(@(h)startsWith(string(h.DisplayName),string(labels{desired(j)})+' (CDW-APFD:'),lines);
  assert(sum(matches)==1);ordered(j)=lines(matches);
 end
 leg=legend(main,ordered,'Location','southeast','FontSize',14,'FontName','Arial');leg.ItemTokenSize=[25 18];leg.Box='off';
 for j=1:7
  assert(startsWith(string(leg.String{j}),string(methods{j})+' (CDW-APFD:'),'Legend order differs from the paper.');
 end
 assert(strcmp(main.XLabel.String,'Inspection Budget (%)'));
 out=fullfile(root,'outputs');if ~isfolder(out),mkdir(out);end
 exportgraphics(fig,fullfile(out,sprintf('Figure_3_Dataset_%d.pdf',d)),'ContentType','vector');close(fig);
end
end
