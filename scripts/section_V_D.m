function section_V_D(datasets, runs)
if nargin < 1, datasets=1:3; end
if nargin < 2, runs=30; end
assert(all(ismember(datasets,1:3)) && runs>=1 && runs==floor(runs));
rows=cell(0,5);
for d=datasets
 [data,y,c,alpha,window,step,root]=paper_data(d);
 r=VoD_Sampler_core(data,alpha,window,step);
 rows(end+1,:)={sprintf('Dataset_%d',d),'max-norm',0,compute_CDWAPFD(r,y,c),'original'}; %#ok<AGROW>
 whole=max(arrayfun(@(x)size(x.Features,1),data));
 r=VoD_Sampler_core(data,alpha,whole,step);
 rows(end+1,:)={sprintf('Dataset_%d',d),'whole trace',0,compute_CDWAPFD(r,y,c),'original'}; %#ok<AGROW>
 r=window_rule_core(data,alpha,window,step,'last');
 rows(end+1,:)={sprintf('Dataset_%d',d),'last window',0,compute_CDWAPFD(r,y,c),'window rule'}; %#ok<AGROW>
 lengths=arrayfun(@(x)size(x.Features,1),data)';
 temp_input=[tempname '.mat']; temp_output=[tempname '.mat'];
 cleanup=onCleanup(@()cleanup_choices(temp_input,temp_output)); %#ok<NASGU>
 save(temp_input,'lengths','window','step','runs');
 generator=fullfile(root,'scripts','window_choices.py');
 [status,message]=system(sprintf('python "%s" "%s" "%s"',generator,temp_input,temp_output));
 assert(status==0,'Window-choice generation failed: %s',message);
 x=load(temp_output,'choices'); assert(isequal(size(x.choices),[numel(data),runs]));
 scores=zeros(runs,1);
 for seed=0:runs-1
  r=window_rule_core(data,alpha,window,step,'uniform',x.choices(:,seed+1));
  scores(seed+1)=compute_CDWAPFD(r,y,c);
  rows(end+1,:)={sprintf('Dataset_%d',d),'uniform window',seed,scores(seed+1),'window rule'}; %#ok<AGROW>
  fprintf('Dataset_%d uniform window seed %d complete.\n',d,seed);
 end
 rows(end+1,:)={sprintf('Dataset_%d',d),'uniform window mean',-1,mean(scores),'window rule'}; %#ok<AGROW>
end
t=cell2table(rows,'VariableNames',{'Dataset','Window','Seed','CDW_APFD','Implementation'});
writetable(t,fullfile(root,'outputs','Section_V_D_windows.csv'));disp(t);
end

function cleanup_choices(input_path,output_path)
if isfile(input_path),delete(input_path);end
if isfile(output_path),delete(output_path);end
end
