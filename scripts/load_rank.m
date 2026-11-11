function rank = load_rank(root,dataset,method,seed)
if strcmp(method,'VoD-Sampler'),seed=0;end
p=fullfile(root,'outputs','rankings',sprintf('Dataset_%d_%s_seed%02d.mat',dataset,method,seed));
if ~isfile(p),error('Missing ranking in outputs/rankings/: generate Dataset_%d %s order %d first.',dataset,method,seed);end
x=load(p,'rank');rank=x.rank(:);
end
