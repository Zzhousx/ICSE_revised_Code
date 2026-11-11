function features=extract_example(dataset)
root=fileparts(fileparts(mfilename('fullpath')));names={'random','Greedy','RL'};
work=fullfile(root,'outputs','examples',sprintf('Dataset_%d',dataset));
source=fullfile(root,'feature extraction',sprintf('extract_feature_%s.m',names{dataset}));
script=fullfile(work,sprintf('extract_feature_%s.m',names{dataset}));copyfile(source,script);
previous=pwd;restore=onCleanup(@()cd(previous)); %#ok<NASGU>
cd(work);addpath(work,'-begin');evalin('base',sprintf('extract_feature_%s',names{dataset}));
if dataset==2,suffix='new33';else,suffix='new3';end
p=fullfile(work,sprintf('DeepScenario_%s_Features_with_Labels_%s.mat',names{dataset},suffix));
x=load(p,'globalFeatures');assert(numel(x.globalFeatures)==1);features=x.globalFeatures{1};
end
