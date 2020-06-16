% ---------------------------------------------------------------------- % 
% Simple bandit task data generation, model fitting, parameter retrieval %
% ---------------------------------------------------------------------- % 
addpath bads-master
addpath fit
addpath plot
addpath simulation

close all 
clear all

% ----------------------------------------------------------------------- %
% DATA GENERATION  
% -----------------------------------------------------------------------% 

% Set parameters
% ----------------------------------------------------------------------- %
% Set seed
seed = 1;
rng(seed)

% colors for plot
blue = [0, 0.4470, 0.7410];
red = [0.6350, 0.0780, 0.1840];

% environment setting
% ------------------------------------------------------------------------%
env_param.nsub = 100;
env_param.nstate = 4;
env_param.naction = 2;
env_param.rewards = [0, 1];
env_param.ntrialperstate = 30;
env_param.ntrial = env_param.ntrialperstate * env_param.nstate;

% option 1 probability of winning points
env_param.p{1} = fliplr(linspace(.6, .9, env_param.nstate));

% option 2 probability of winning points
env_param.p{2} = 1 - env_param.p{1};

env_param.ev{1} = 1*env_param.p{1} + -1*(1-env_param.p{1});
env_param.ev{2} = 1*env_param.p{2} + -1*(1-env_param.p{2});
env_param.con = repelem(1:env_param.nstate, env_param.ntrialperstate);

% model setting
% ------------------------------------------------------------------------%
model_param.nstate = env_param.nstate;
model_param.naction = env_param.naction;

% qvalue initialization
model_param.initq = .5;
% decision rule: 1=softmax, 2=argmax
model_param.decision_rule = 1;
% counterfactual learning?
model_param.counterfactual = 1;
% generate random 'cognitive' parameters
model_param.beta_dist = [1, 1];
model_param.gam_dist = [1.2, 5];
model_param.lr = betarnd(model_param.beta_dist(1),...
                         model_param.beta_dist(2), [env_param.nsub, 1]);
model_param.temp = gamrnd(model_param.gam_dist(1),...
                         model_param.gam_dist(2), [env_param.nsub, 1]);
                     
% Simulate and plot the learnt values
% ------------------------------------------------------------------------%
[env_param.cho, env_param.out,  env_param.cfcho, env_param.cfout, Q] = ...
    run_simulation_test(model_param, env_param);
% reorganize qvalues in order to plot them properly
Q = sort_Q(Q);

% plot Q-values to check learnt values from agents
brickplot(Q', ...
    [0, 0.4470, 0.7410] .* ones(8, 3),...
    [0, 1],...
    [0, 1],...
    15,...
    '',...
    'P(win)', 'Estimated P(win)',...
    sort([env_param.p{2} env_param.p{1}]),...
    sort([env_param.p{2} env_param.p{1}])...
);
plot(linspace(-1, 1, 10), zeros(10, 1), 'linestyle', ':', 'color', 'k');
box off
set(gca, 'tickdir', 'out');


% ---------------------------------------------------------------------- % 
% PARAMETER FITTING AND RETRIEVAL
% -----------------------------------------------------------------------% 
fit_param = env_param;
fit_param.counterfactual = model_param.counterfactual;
fit_param.decision_rule = model_param.decision_rule;
fit_param.initq = model_param.initq;
fit_param.init_value = [1, .5];
fit_param.lb = [0.01, 0.01];
fit_param.ub = [100, 1];
fit_param.beta_dist = model_param.beta_dist;
fit_param.gam_dist = model_param.gam_dist;

% 1 = optimize the log likelihood, 0 the likelihood
fit_param.logLL = 1;


% FMINCON
% ----------------------------------------------------------------------- %
[params_fmincon, lpp_fmincon] = runfit(fit_param, 'fmincon');

figure('position', [682,270,1000,400]);

subplot(1, 2, 1);
scatterplot(...
    model_param.temp, params_fmincon(:, 1), blue, [0, 20], [0, 20],...
    'Real value', 'Fitted value', '\beta');

subplot(1, 2, 2);
scatterplot(...
    model_param.lr, params_fmincon(:, 2), red, [0, 1], [0, 1],...
    'Real value', 'Fitted value', '\alpha');

suptitle('FMINCON');


% BADS
% ----------------------------------------------------------------------- %
[params_bads, lpp_bads] = runfit(fit_param, 'bads');

figure('position', [682,270,1000,400]);

subplot(1, 2, 1);
scatterplot(...
    model_param.temp, params_bads(:, 1), blue, [0, 20], [0, 20],...
    'Real value', 'Fitted value', '\beta');


subplot(1, 2, 2);
scatterplot(...
    model_param.lr, params_bads(:, 2), red, [0, 1], [0, 1],...
    'Real value', 'Fitted value', '\alpha');

suptitle('BADS');

% -----------------------------------------------------------------------% 
% SIDE FUNCTIONS                                                              
% -----------------------------------------------------------------------%
function new_Q = sort_Q(Q)
    map = [2 4 6 8 7 5 3 1];

    for i = 1:size(Q, 1)

        t_Q(1:size(Q,2), 1:2) = Q(i, :, :);

        new_Q(i, :) = reshape(t_Q', [], 1);
        new_Q(i, :) = new_Q(i, map);
    end
end


