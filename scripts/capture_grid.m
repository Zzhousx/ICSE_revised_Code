function stop=capture_grid
stop=false;values=getappdata(groot,'vod_sensitivity_grid');
assignin('caller','acc_data_orl',values*100);
end
