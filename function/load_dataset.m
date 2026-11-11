function data = load_dataset(datasetPath)
    if ~isfile(datasetPath)
        error('Cannot find the dataset file: %s', datasetPath);
    end
    raw = load(datasetPath);
    dt  = 0.5;
    N   = length(raw.globalFeatures);

    data = struct('Features', cell(N,1), 'Label', cell(N,1), ...
                  'Duration', cell(N,1), 'ClassID', cell(N,1));
    for i = 1:N
        data(i).Features = raw.globalFeatures{i};
        data(i).Label    = raw.globalLabels(i);
        data(i).ClassID  = raw.globalClassIDs(i);
        T = size(raw.globalFeatures{i}, 1);
        data(i).Duration = max(T * dt, 0.1);
    end
    
    validIdx = arrayfun(@(x) size(x.Features,2)==18 && ~isempty(x.Features), data);
    data = data(validIdx);
end