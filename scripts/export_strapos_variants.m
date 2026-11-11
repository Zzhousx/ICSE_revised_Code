function export_strapos_variants
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'scripts'),fullfile(root,'function'));
out=fullfile(root,'outputs');
src=fullfile(out,'fullrank');
configs={'main','semantic_coverage','time_equivalent','distance_30','distance_50','distance_100','pedestrian_proxy','joined_recordings','single_pass'};
if ~isfolder(out),mkdir(out);end
f=fopen(fullfile(out,'variant_metrics.csv'),'w');
assert(f~=-1,'Cannot write outputs/variant_metrics.csv.');
closeFile=onCleanup(@() fclose(f));
fprintf(f,'configuration,dataset,CDW_APFD_mean,CDW_APFD_std,FD_5_mean,FD_5_std,CCR_5_T68_mean,CCR_5_T68_std,CCR_5_T17_mean,CCR_5_T17_std,unranked_mean\n');
for j=1:numel(configs)
    for d=1:3
        [~,y,c,~,~,~,~]=paper_data(d);
        c17=c;c17(c>0)=mod(c(c>0)-1,17)+1;
        configDir=fullfile(src,configs{j},sprintf('D%d',d));
        diagPath=fullfile(configDir,'diagnostics.csv');
        assert(isfile(diagPath), ...
            'Missing %s. Generate main rankings and run python scripts/rank_strapos_variants.py first.',diagPath);
        opts=detectImportOptions(diagPath,'FileType','text','Delimiter',',', ...
            'VariableNamesLine',1,'VariableNamingRule','preserve');
        opts.DataLines=[2 Inf];
        if ismember('unranked_logs',opts.VariableNames)
            unrankedField='unranked_logs';
        elseif ismember('tier_C_logs',opts.VariableNames)
            unrankedField='tier_C_logs';
        else
            error('Missing unranked_logs or tier_C_logs in %s.',diagPath);
        end
        opts.SelectedVariableNames={'seed',unrankedField};
        opts=setvartype(opts,opts.SelectedVariableNames,'double');
        diagnostics=readtable(diagPath,opts);
        assert(height(diagnostics)==30 && isequal(sort(diagnostics.seed),(0:29)'), ...
            'Expected seeds 0-29 in %s.',diagPath);
        unranked=diagnostics.(unrankedField);
        v=zeros(30,4);
        for seed=0:29
            rankPath=fullfile(configDir,sprintf('ranking_seed%02d.csv',seed));
            assert(isfile(rankPath),'Missing generated ranking: %s.',rankPath);
            r=double(readmatrix(rankPath));r=r(:);
            assert(numel(r)==numel(y) && isequal(sort(r),(1:numel(y))'), ...
                'Not a complete ranking of Dataset_%d: %s.',d,rankPath);
            v(seed+1,:)=[compute_CDWAPFD(r,y,c),compute_FaultCurve(r,y,5)/100, ...
                compute_CCR(r,y,c,.05,unique(c(y==1 & c>0))), ...
                compute_CCR(r,y,c17,.05,unique(c17(y==1 & c17>0)))];
        end
        name=configs{j};
        if strcmp(name,'pedestrian_proxy')
            name='pedestrian_proxy (proxy, not STRaP''s definition)';
        elseif strcmp(name,'single_pass')
            name='single pass, rest by score';
        end
        fprintf(f,'"%s",%d',name,d);
        for k=1:4
            fprintf(f,',%.17g,%.17g',mean(v(:,k)),std(v(:,k)));
        end
        fprintf(f,',%.17g\n',mean(unranked));
    end
end
end
