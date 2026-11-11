function preprocess(dataset)
root=fileparts(fileparts(mfilename('fullpath')));
assert(ismember(dataset,1:3));
names={'random','Greedy','RL'};
work=fullfile(root,'outputs','preprocessing',sprintf('Dataset_%d',dataset));
if ~isfolder(work),mkdir(work);end
dest=fullfile(work,'deepscenario-dataset');if ~isfolder(dest),mkdir(dest);end
unzip(fullfile(root,'data',sprintf('Dataset_%d_raw.zip',dataset)),dest);
source=fullfile(root,'feature extraction',sprintf('extract_feature_%s.m',names{dataset}));
script=fullfile(work,sprintf('extract_feature_%s.m',names{dataset}));copyfile(source,script);
previous=pwd;restore=onCleanup(@()cd(previous)); %#ok<NASGU>
cd(work);
addpath(work,'-begin');command=sprintf('extract_feature_%s',names{dataset});
evalin('base',command);
fprintf('Dataset_%d extracted data are in outputs/preprocessing/.\n',dataset);
end
