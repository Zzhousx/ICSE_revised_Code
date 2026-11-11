function walkthrough
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'scripts'),fullfile(root,'walkthrough'));
ids={'random-strategy/reward-dto/road1-rain_day-scenarios/2_scenario_7_pedestrian.deepscenario', ...
 'greedy-strategy/reward-jerk/road3-sunny_night-scenarios/0_scenario_9_npc_vehicle.deepscenario', ...
 'rl_based-strategy/reward-dto/road4-sunny_night-scenarios/15_scenario_4_npc_vehicle.deepscenario'};
targets=[38 7634 1923]; ranks=[53 54 25];
if ~isfolder(fullfile(root,'outputs')),mkdir(fullfile(root,'outputs'));end
[status,message]=system(sprintf('python "%s"',fullfile(root,'walkthrough','prepare_examples.py')));
if status~=0,error('Unable to prepare recorded logs: %s',message);end
for d=1:3
 [data,~,~,alpha,window,step]=paper_data(d);target=targets(d);
 extracted=extract_example(d);
 assert(isequaln(extracted,data(target).Features),'Extracted features differ from the benchmark input.');
 assignin('base','vod_walkthrough_target',target);
 dbstop in VoD_Sampler_core at 88 if capture_window()
 dbstop in feature_space_greedy_dpp at 35 if capture_basis()
 dbstop in feature_space_greedy_dpp at 43 if capture_bucket()
 [rank,V,q]=VoD_Sampler_core(data,alpha,window,step);
 dbclear in VoD_Sampler_core;dbclear in feature_space_greedy_dpp;
 assert(rank(ranks(d))==target,'The requested log does not have the expected rank.');
 x=evalin('base','vod_walkthrough_window');basis=evalin('base','vod_walkthrough_basis');bucket=evalin('base','vod_walkthrough_bucket');
 fprintf('\nDataset_%d log ID: %s\n',d,ids{d});
 fprintf('The 18 input features by frame:\n');disp(data(target).Features);
 fprintf('Window start, end and intensity:\n');disp([x.starts(:),x.ends(:),x.intensities(:)]);
 fprintf('Selected window: %d, frames %d to %d.\n',x.selected,x.starts(x.selected),x.ends(x.selected));
 fprintf('Covariance, obtained as expm of the selected log-covariance:\n');disp(expm(x.log_covariance));
 fprintf('Normalized tangent vector:\n');disp(V(target,:));
 fprintf('Quality q = %.17g; basis member = %d; bucket = %d; bucket anchor = %d; other logs in bucket = %d; final rank = %d.\n', ...
  q(target),ismember(target,basis),bucket.index,bucket.anchor,bucket.other_logs,ranks(d));
end
end
