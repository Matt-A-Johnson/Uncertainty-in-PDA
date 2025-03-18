function lpp = getlpp_learning(params, s, a, cfa, r, cfr, q, ntrials, decision_rule, fit_cf)
addpath './'

%%% log prior of parameters
beta1 = params(1); % choice temperature
alpha1 = params(2); % policy or factual learning rate
pbeta1 = log(gampdf(beta1, 1.2, 5.0));
palpha1 = log(betapdf(alpha1, 1.1, 1.1));
p = -sum([pbeta1, palpha1]);

model = QLearning(params, q, 2, 2, ntrials, decision_rule);
l = model.fit(s, a, cfa, r, cfr, fit_cf);

lpp = p + l;
end



