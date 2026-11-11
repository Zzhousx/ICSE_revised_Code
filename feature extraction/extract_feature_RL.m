% =========================================================================
% DeepScenario Feature Extractor -RL
% =========================================================================
clear; clc;

% 1. Configuration path
rootDir = '.\deepscenario-dataset\rl_based-strategy'; 
dt = 0.5; 


fprintf('Scanning the directory: %s\n', rootDir);
csvFiles = dir(fullfile(rootDir, '**', '*-attributes.csv'));

if isempty(csvFiles)
    error('No CSV files were found! Please check the path.');
end

MAX_SIZE = 15000;
globalFeatures = cell(MAX_SIZE, 1);
globalLabels = zeros(MAX_SIZE, 1);               
globalMetadata = strings(MAX_SIZE, 1);           
globalTextLabels = strings(MAX_SIZE, 1);         
globalClassIDs = zeros(MAX_SIZE, 1);            
counter = 0;

for f = 1:length(csvFiles)
    folderPath = csvFiles(f).folder; 
    csvPath = fullfile(folderPath, csvFiles(f).name);
    
    fprintf('being analyzed: %s\n', csvFiles(f).name);
    opts = detectImportOptions(csvPath, 'VariableNamingRule', 'preserve');
    scenarioData = readtable(csvPath, opts);
    isRain = contains(folderPath, 'rain', 'IgnoreCase', true);
    isNight = contains(folderPath, 'night', 'IgnoreCase', true);
    frictionCoeff = 1.0; if isRain, frictionCoeff = 0.6; end
    timeOfDaySin = sin(2*pi*12/24); if isNight, timeOfDaySin = sin(2*pi*20/24); end
    
    for row = 1:height(scenarioData)
        scenarioID = string(scenarioData.ScenarioID{row});
        colColName = 'Attribute[COL]';
        colVal = scenarioData.(colColName)(row);
        isCollision = islogical(colVal) && colVal || strcmpi(string(colVal), 'True');
        xmlFile = fullfile(folderPath, strcat(scenarioID, '.deepscenario'));
        xmlText = fileread(xmlFile);
        
        [egoVx, egoVz, egoPx, egoPz, egoYawRate, egoYaw] = extractTimeSeries(xmlText, 'Act_Ego0');
        if isempty(egoVx) || length(egoVx) < 2
            continue; 
        end
        N = length(egoVx);
   
        % =========== Scenario Classification ===========
        desc = lower(string(scenarioData.Configuration_API_Description{row}));
        
        % 1. Determining the environmental offset depends on isRain and isNight
        envOffset = 0;
        envStr = "";
        if ~isRain && ~isNight
            envOffset = 0;   envStr = "Sunny_Day";
        elseif ~isRain && isNight
            envOffset = 17;  envStr = "Sunny_Night";
        elseif isRain && ~isNight
            envOffset = 34;  envStr = "Rain_Day";
        elseif isRain && isNight
            envOffset = 51;  envStr = "Rain_Night";
        end

        % 2. Extract 17 basic scenario categories
        baseClass = 0;
        baseText = "";
        
        if contains(desc, 'pedestrian')
            baseClass = 17;
            baseText = "Pedestrian_Crossing";
        else
            % Vehicle scenarios (1 to 16 categories)
            % Dimension A: Topological Relationship (0-3)
            posIdx = 0; posStr = "Unknown";
            if contains(desc, 'driving ahead')
                posIdx = 0; posStr = "Ahead";
            elseif contains(desc, 'overtaking')
                posIdx = 1; posStr = "Overtaking";
            elseif contains(desc, 'opposite')
                posIdx = 2; posStr = "Opposite";
            elseif contains(desc, 'crossing')
                posIdx = 3; posStr = "Crossing";
            end
            
            % Dimension B: Initial Distance (0 - 1)
            distIdx = 0; distStr = "Unknown";
            if contains(desc, '(near)')
                distIdx = 0; distStr = "Near";
            elseif contains(desc, '(far)')
                distIdx = 1; distStr = "Far";
            end
            
            % Dimension C: Lateral Movement (0-1)
            actIdx = 0; actStr = "Unknown";
            if contains(desc, 'maintaining')
                actIdx = 0; actStr = "Maintaining";
            elseif contains(desc, 'switching')
                actIdx = 1; actStr = "Switching";
            end
            
            % Binary cascading calculation BaseClass
            baseClass = posIdx * 4 + distIdx * 2 + actIdx + 1;
            baseText = sprintf("Vehicle_%s_%s_%s", posStr, distStr, actStr);
        end
        
        % 3. Merge to obtain the final 1-68 categories
        classID = baseClass + envOffset;
        textLabel = baseText + "_" + envStr;
        %=========================================
        agentIds = regexp(xmlText, '<ObjectAction name="(Act_[^"]+)">', 'tokens');
        bestDist = inf;
        bestNpcVx = zeros(N,1); bestNpcVz = zeros(N,1); 
        bestNpcPx = zeros(N,1); bestNpcPz = zeros(N,1);

        if ~isempty(agentIds)
            for i = 1:length(agentIds)
                agentName = agentIds{i}{1};
                if contains(agentName, 'Ego')
                    continue; 
                end
                [tempVx, tempVz, tempPx, tempPz, ~, ~] = extractTimeSeries(xmlText, agentName);
                
                if length(tempVx) == N
                    distSeq = sqrt((tempPx - egoPx).^2 + (tempPz - egoPz).^2);
                    minDistForThisAgent = min(distSeq);
                    if minDistForThisAgent < bestDist
                        bestDist = minDistForThisAgent;
                        bestNpcVx = tempVx; bestNpcVz = tempVz;
                        bestNpcPx = tempPx; bestNpcPz = tempPz;
                    end
                end
            end
        end
        npcVx = bestNpcVx; npcVz = bestNpcVz; 
        npcPx = bestNpcPx; npcPz = bestNpcPz;
        
        % =========== Feature calculation and assembly ===========

        egoAx = calcDeriv(egoVx, dt); egoAz = calcDeriv(egoVz, dt);
        egoJx = calcDeriv(egoAx, dt); egoJz = calcDeriv(egoAz, dt);
        egoYawAccel = calcDeriv(egoYawRate, dt);
        egoYawCos = cosd(egoYaw); egoYawSin = sind(egoYaw);
        
        relDistX = npcPx - egoPx; 
        relDistZ = npcPz - egoPz;
        relVelX  = npcVx - egoVx; 
        relVelZ  = npcVz - egoVz; 
        relDist = sqrt(relDistX.^2 + relDistZ.^2);
        proxPotential = exp(-relDist / 10.0); 
        approachRate = -(relDistX .* relVelX + relDistZ .* relVelZ) ./ (relDist + 1e-5);
        invTTC = max(0, approachRate) ./ (relDist + 1e-5);
        invTTC(relDist > 50) = 0; 
        npcAx = calcDeriv(npcVx, dt); 
        npcAz = calcDeriv(npcVz, dt);
        featMat =[egoVx, egoVz, egoAx, egoAz, egoJx, egoJz, ...
                  egoYawCos, egoYawSin, egoYawRate, egoYawAccel, ...
                  proxPotential, invTTC, relVelX, relVelZ, npcAx, npcAz, ...
                  ones(N,1)*frictionCoeff, ones(N,1)*timeOfDaySin];
        
        % =========== Store in the global variable ===========
        counter = counter + 1;
        globalFeatures{counter} = featMat;
        globalLabels(counter) = isCollision;
        globalMetadata(counter) = scenarioID;
        globalTextLabels(counter) = textLabel;
        globalClassIDs(counter) = classID;
    end
end

% =========== Result storage ===========
globalFeatures = globalFeatures(1:counter);
globalLabels = globalLabels(1:counter);
globalMetadata = globalMetadata(1:counter);
globalTextLabels = globalTextLabels(1:counter);
globalClassIDs = globalClassIDs(1:counter);

save('DeepScenario_RL_Features_with_Labels_new3.mat', ...
     'globalFeatures', 'globalLabels', 'globalMetadata', ...
     'globalTextLabels', 'globalClassIDs', '-v7.3');

fprintf('\n========= Extract completed =========\n');
fprintf('Effective scene features have been aligned with their corresponding classifications! A total of %d use cases.\n', counter);

ResultTable = table(globalMetadata, globalTextLabels, globalClassIDs, 'VariableNames', {'ID', 'TextLabel', 'ClassID'});
summaryStats = groupsummary(ResultTable, {'ClassID', 'TextLabel'});
summaryStats = sortrows(summaryStats, 'GroupCount', 'descend');
disp(summaryStats(:, {'ClassID', 'TextLabel', 'GroupCount'}));


function d = calcDeriv(x, dt)
    d = zeros(size(x));
    d(1:end-1) = diff(x) / dt;
    d(end) = d(end-1);
end

function[vx, vz, px, pz, yawrate, yaw] = extractTimeSeries(xmlText, actName)
    idx = strfind(xmlText, sprintf('name="%s"', actName));
    if isempty(idx), vx=[]; vz=[]; px=[]; pz=[]; yawrate=[]; yaw=[]; return; end
    
    actBlock = xmlText(idx(1):end);
    nextAct = strfind(actBlock, '<ObjectAction name="');
    if length(nextAct) > 1, actBlock = actBlock(1:nextAct(2)-1); end
    
    vx = getMatches(actBlock, 'velocityX="([^"]+)"');
    vz = getMatches(actBlock, 'velocityZ="([^"]+)"');
    px = getMatches(actBlock, 'positionX="([^"]+)"');
    pz = getMatches(actBlock, 'positionZ="([^"]+)"');
    yawrate = getMatches(actBlock, 'angularVelocityY="([^"]+)"');
    yaw = getMatches(actBlock, 'rotationY="([^"]+)"');
end

function vals = getMatches(textBlock, pattern)
    tokens = regexp(textBlock, pattern, 'tokens');
    vals = zeros(length(tokens), 1);
    for k = 1:length(tokens), vals(k) = str2double(tokens{k}{1}); end
end