function table_IV_summary
root=fileparts(fileparts(mfilename('fullpath')));
t=readtable(fullfile(root,'outputs','Table_IV_runs.csv'),'TextType','string');
for d=1:3
 p=fullfile(root,'outputs',sprintf('Table_IV_STRaP-OS_Dataset_%d.csv',d));
 os=readtable(p,'TextType','string');t=[t;os]; %#ok<AGROW>
end
u=groupsummary(t,{'Dataset','Method'},{'median','mean'},'Seconds');
u.Method=categorical(u.Method,{'VoD-Sampler','SO-SDC','MO-SDC','STRaP-IR','STRaP-OS','BehAVExplor'},'Ordinal',true);
u=sortrows(u,{'Dataset','Method'});
writetable(u,fullfile(root,'outputs','Table_IV.csv'));disp(u);
end
