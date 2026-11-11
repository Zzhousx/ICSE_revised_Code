function stop = capture_window
stop=false;
if evalin('caller','i') ~= evalin('base','vod_walkthrough_target'),return;end
x=struct();
x.starts=evalin('caller','wStarts');x.ends=evalin('caller','wEnds');
x.intensities=evalin('caller','temp_intens');x.selected=evalin('caller','max_idx');
x.log_covariance=evalin('caller','temp_S{max_idx}');
assignin('base','vod_walkthrough_window',x);
end
