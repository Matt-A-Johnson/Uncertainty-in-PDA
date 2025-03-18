clearvars; close all;

allData = readtable('LM_Data.csv');

condnames = unique(allData.blockLabel);
allsubs = unique(allData.subject);
numsubs = length(allsubs);
nll = zeros(2,numsubs);
parameters = zeros(2,2,numsubs); % 2 temperature (b), 2 alpha (stable, volatile) for each participant
% probwin = zeros(6,numsubs);

options = optimset('Algorithm','interior-point','Display', 'off',...
    'MaxIter', 10000,'MaxFunEval', 10000);

% fmincon params
fmincon_params.init_value = [0, .5, .5];
fmincon_params.lb = [0.001, 0.001, 0.001];
fmincon_params.ub = [100, 1, 1];

for currsub = 1:numsubs
    subdata = allData(strcmp(allData.subject,allsubs(currsub)),:);
    for currcond = 1:length(condnames)
        % Filter data by condition and select trials >=21
        condData = subdata(strcmp(subdata.blockLabel, condnames{currcond}) & ...
                           subdata.blockCount >= 21, :);
        
        % Debug: Display subject and condition details
        disp(['Subject ' num2str(currsub) ', Condition: ' condnames{currcond} ', Trial Count: ' num2str(height(condData))]);
        
        % Set condition column for analysis
        condData.condition = repelem(currcond,height(condData),1);

        % set parameters
        fit_params.cho = condData.b_correct+1;
        fit_params.cfcho = 2-condData.b_correct;
        fit_params.out = double(condData.feedback>0);
        fit_params.cfout = double(condData.feedback<0);
        fit_params.con = condData.condition;
        fit_params.fit_cf = 0;
        fit_params.ntrials = height(condData);
        fit_params.model = 1;
        fit_params.decision_rule = 1;
        fit_params.q = 0.5;
        fit_params.noptions = 2;
        fit_params.ncond = 1;

        [params, ll] = runfit_learning(fit_params, fmincon_params, options);
        parameters(:,currcond,currsub) = params(1:2);
        nll(currcond,currsub)=ll;
    end

end

% Prepare data for the DataFrame
subjectIDs = allsubs; % Create a column vector of subject IDs
stable_alpha = squeeze(parameters(2, 1, :)); % Extract stable alpha values
stable_beta = squeeze(parameters(1, 1, :));  % Extract stable beta values
volatile_alpha = squeeze(parameters(2, 2, :)); % Extract volatile alpha values
volatile_beta = squeeze(parameters(1, 2, :));  % Extract volatile beta values

% Ensure all variables are column vectors
subjectIDs = subjectIDs(:);  % Ensure subjectIDs is a column vector
stable_alpha = stable_alpha(:);  % Ensure stable_alpha is a column vector
stable_beta = stable_beta(:);  % Ensure stable_beta is a column vector
volatile_alpha = volatile_alpha(:);  % Ensure volatile_alpha is a column vector
volatile_beta = volatile_beta(:);  % Ensure volatile_beta is a column vector

% Create a table
resultsTable = table(subjectIDs, stable_alpha, stable_beta, volatile_alpha, volatile_beta, ...
                     'VariableNames', {'subject', 'stableAlpha', 'stableBeta', 'volatileAlpha', 'volatileBeta'});

% Save the table as a .csv file or any other format as needed
writetable(resultsTable, 'results_RL_summary_21_80.csv');

% Optionally, display the table
disp(resultsTable)

save results_RL_summary_21_80.mat parameters nll

function [parameters,ll] = runfit_learning(fit_params, fmincon_params, options)
[parameters,l1] = fmincon(@(x) getlpp_learning(x,...
    fit_params.con,...
    fit_params.cho,...
    fit_params.cfcho,...
    fit_params.out,...
    fit_params.cfout,...
    fit_params.q,...
    fit_params.ntrials, fit_params.decision_rule,fit_params.fit_cf),...
    fmincon_params.init_value,...
    [], [], [], [],...
    fmincon_params.lb,...
    fmincon_params.ub,...
    [],options);
ll = l1;
end