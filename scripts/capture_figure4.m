function stop=capture_figure4
stop=false; os=getappdata(groot,'vod_figure4_os');
a=evalin('caller','data_D1');b=evalin('caller','data_D2');c=evalin('caller','data_D3');
dat={b,a,c};
for d=1:3, dat{d}=[dat{d}(:,1:4),os{d},dat{d}(:,5:6)]; end
all_data=[dat{1};dat{2};dat{3}];
colors=evalin('caller','colors_all');
colors=[colors(1:4,:);145/255 209/255 194/255;colors(5:6,:)];
assignin('caller','data_All',all_data);
assignin('caller','datasets_data',{dat{1},dat{2},dat{3},all_data});
assignin('caller','methods',{'VoD-Sampler','SO-SDC','MO-SDC','STRaP-IR','STRaP-OS','BehAVExplor','Random'});
assignin('caller','num_methods',7);assignin('caller','colors_all',colors);
end
