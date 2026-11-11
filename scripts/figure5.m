function figure5
alpha=[.1 .2 .3 .4 .5];windows=3:7;rows=cell(0,4);
for d=1:3
 [data,y,c,~,~,step,root]=paper_data(d);values=zeros(numel(alpha),numel(windows));
 for i=1:numel(alpha)
  for j=1:numel(windows)
   rank=VoD_Sampler_core(data,alpha(i),windows(j),step);
   values(i,j)=compute_CDWAPFD(rank,y,c);
   rows(end+1,:)={sprintf('Dataset_%d',d),alpha(i),windows(j),values(i,j)}; %#ok<AGROW>
   fprintf('Dataset_%d alpha %.1f window %d complete.\n',d,alpha(i),windows(j));
  end
 end
 setappdata(groot,'vod_sensitivity_grid',values);
 dbstop in Robustness at 16 if capture_grid()
 evalin('base','Robustness');dbclear in Robustness
 fig=evalin('base','gcf');
 exportgraphics(fig,fullfile(root,'outputs',sprintf('Figure_5_Dataset_%d.pdf',d)),'ContentType','vector');close(fig);
end
t=cell2table(rows,'VariableNames',{'Dataset','Alpha','Window','CDW_APFD'});
writetable(t,fullfile(root,'outputs','Figure_5_data.csv'));disp(t);
end
