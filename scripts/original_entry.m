function original_entry(entry)
assert(ismember(string(entry),["run","ablation"]));
root=fileparts(fileparts(mfilename('fullpath')));
work=fullfile(root,'outputs','author_entry');if ~isfolder(work),mkdir(work);end
for name={'run.m','ablation.m','main_evaluation.m','VoD_Sampler_core.m','VoD_Sampler_ablation.m'}
 copyfile(fullfile(root,name{1}),fullfile(work,name{1}));
end
for folder={'function','experiment'}
 copyfile(fullfile(root,folder{1}),fullfile(work,folder{1}));
end
data_dir=fullfile(work,'feature extraction');if ~isfolder(data_dir),mkdir(data_dir);end
names={'random','Greedy','RL'};
for d=1:3
 copyfile(fullfile(root,'data',sprintf('Dataset_%d.mat',d)), ...
  fullfile(data_dir,sprintf('DeepScenario_%s_Features_with_Labels_new3.mat',names{d})));
end
previous=pwd;restore=onCleanup(@()cd(previous)); %#ok<NASGU>
cd(work);addpath(work,'-begin');evalin('base',char(entry));
end
