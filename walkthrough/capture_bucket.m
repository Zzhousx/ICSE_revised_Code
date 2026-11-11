function stop = capture_bucket
stop=false;
anchors=evalin('caller','Y_out');assignment=evalin('caller','bucket_assignment');
target=evalin('base','vod_walkthrough_target');bucket=find(anchors==target,1);
if isempty(bucket)
 remaining=evalin('caller','rem_idx'); k=find(remaining==target,1);
 if isempty(k),error('The selected log is absent from the bucket assignment.');end
 bucket=assignment(k);
end
assignin('base','vod_walkthrough_bucket',struct('index',bucket,'anchor',anchors(bucket),'other_logs',sum(assignment==bucket)));
end
