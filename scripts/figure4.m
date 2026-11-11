function figure4
for d=1:3
 [~,y,c,~,~,~,root]=paper_data(d); scores=zeros(30,1);
 for seed=0:29
  rank=load_rank(root,d,'STRaP-OS',seed);
  scores(seed+1)=compute_CDWAPFD(rank,y,c);
 end
 os{d}=scores; %#ok<AGROW>
end
setappdata(groot,'vod_figure4_os',os);
cleanup=onCleanup(@()clear_figure4()); %#ok<NASGU>
dbstop in boxline at 53 if capture_figure4()
evalin('base','boxline');
fig=evalin('base','fig');
exportgraphics(fig,fullfile(root,'outputs','Figure_4.pdf'),'ContentType','vector');
close(fig);
end
function clear_figure4
dbclear in boxline
if isappdata(groot,'vod_figure4_os'),rmappdata(groot,'vod_figure4_os');end
end
