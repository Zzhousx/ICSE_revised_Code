function recompute_own_budget
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'scripts'),fullfile(root,'function'));
out=fullfile(root,'outputs');
if ~isfolder(out),mkdir(out);end
f=fopen(fullfile(out,'own_budget_runs.csv'),'w');
assert(f~=-1,'Cannot write outputs/own_budget_runs.csv.');
closeRuns=onCleanup(@() fclose(f));
fprintf(f,'dataset,method,seed,n1,FD,CCR68,CCR17\n');
stats=fopen(fullfile(out,'statistics.csv'),'w');
assert(stats~=-1,'Cannot write outputs/statistics.csv.');
closeStats=onCleanup(@() fclose(stats));
fprintf(stats,'dataset,metric,A12,p,n\n');
for d=1:3
    [~,y,c,~,~,~,~]=paper_data(d);
    c17=c;c17(c>0)=mod(c(c>0)-1,17)+1;N=numel(y);
    configDir=fullfile(out,'fullrank','main',sprintf('D%d',d));
    diagPath=fullfile(configDir,'diagnostics.csv');
    assert(isfile(diagPath), ...
        'Missing %s. Run python scripts/rank_full.py --dataset %d first.',diagPath,d);
    opts=detectImportOptions(diagPath,'FileType','text','Delimiter',',', ...
        'VariableNamesLine',1,'VariableNamingRule','preserve');
    opts.DataLines=[2 Inf];
    assert(ismember('round1_logs',opts.VariableNames), ...
        'Missing round1_logs in %s.',diagPath);
    opts.SelectedVariableNames={'seed','round1_logs'};
    opts=setvartype(opts,opts.SelectedVariableNames,'double');
    diagnostics=readtable(diagPath,opts);
    assert(height(diagnostics)==30 && isequal(sort(diagnostics.seed),(0:29)'), ...
        'Expected seeds 0-29 in %s.',diagPath);
    assert(numel(unique(diagnostics.round1_logs))==1, ...
        'First-round log counts differ across seeds in %s.',diagPath);
    n1=double(diagnostics.round1_logs(1));
    assert(n1>=1 && n1<=N && n1==round(n1),'Invalid first-round budget in %s.',diagPath);
    ranks=zeros(N,30);main=zeros(30,4);
    for seed=0:29
        rankPath=fullfile(configDir,sprintf('ranking_seed%02d.csv',seed));
        assert(isfile(rankPath),'Missing generated ranking: %s.',rankPath);
        r=double(readmatrix(rankPath));r=r(:);
        assert(numel(r)==N && isequal(sort(r),(1:N)'), ...
            'Not a complete ranking of Dataset_%d: %s.',d,rankPath);
        ranks(:,seed+1)=r;
        main(seed+1,:)=[compute_CDWAPFD(r,y,c),compute_FaultCurve(r,y,5)/100, ...
            compute_CCR(r,y,c,.05,unique(c(y==1 & c>0))), ...
            compute_CCR(r,y,c17,.05,unique(c17(y==1 & c17>0)))];
    end
    vodRank=load_rank(root,d,'VoD-Sampler',0);
    vod=[compute_CDWAPFD(vodRank,y,c),compute_FaultCurve(vodRank,y,5)/100, ...
        compute_CCR(vodRank,y,c,.05,unique(c(y==1 & c>0))), ...
        compute_CCR(vodRank,y,c17,.05,unique(c17(y==1 & c17>0)))];
    labels={'CDW-APFD','FD@5','CCR@5 T68','CCR@5 T17'};
    for k=1:4
        [v,p]=compute_StatTest(vod(k),main(:,k));
        fprintf(stats,'%d,%s,%.17g,%.17g,30\n',d,labels{k},v,p);
    end
    names={'STRaP-OS','VoD-Sampler','Random'};
    for m=1:3
        for seed=0:29
            if m==1
                r=ranks(:,seed+1);
            elseif m==2
                r=vodRank;
            else
                r=load_rank(root,d,'Random',seed);
            end
            fprintf(f,'%d,%s,%d,%d,%.17g,%.17g,%.17g\n',d,names{m},seed,n1, ...
                compute_FaultCurve(r,y,100*n1/N)/100, ...
                compute_CCR(r,y,c,n1/N,unique(c(y==1 & c>0))), ...
                compute_CCR(r,y,c17,n1/N,unique(c17(y==1 & c17>0))));
        end
    end
end
end
