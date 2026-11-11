function section_V_C
rows=cell(0,3);
for d=1:3
 [data,~,~,alpha,window,step,root]=paper_data(d);addpath(fullfile(root,'walkthrough'));
 dbstop in feature_space_greedy_dpp at 35 if capture_basis()
 [rank,V,q]=VoD_Sampler_core(data,alpha,window,step);
 dbclear in feature_space_greedy_dpp
 basis=evalin('base','vod_walkthrough_basis');K=round(.05*numel(rank));
 share=sum(ismember(rank(1:K),basis))/K;
 out=fullfile(root,'outputs','rankings');if ~isfolder(out),mkdir(out);end
 seed=0;dataset=d;save(fullfile(out,sprintf('Dataset_%d_VoD-Sampler_seed00.mat',d)),'rank','V','q','seed','dataset');
 rows(end+1,:)={sprintf('Dataset_%d',d),numel(basis),share}; %#ok<AGROW>
end
t=cell2table(rows,'VariableNames',{'Dataset','BasisSize','BasisShare5'});
writetable(t,fullfile(root,'outputs','Section_V_C_basis.csv'));disp(t);
end
