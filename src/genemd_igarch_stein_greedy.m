%%
% File: genemd_igarch_stein_greedy.m
% Purpose:
% Generates EMD plots for various KSD greedy algorithms. The target is the
% posterior of an IGARCH model.
% Date: February 8, 2018
%%

% Number of particles
nPart = 100;

% Load S&P 500 return data
load('data_spx.mat');
str = 2501;
wdt = 2000;
r = r(str:(str + wdt - 1));

% GARCH score function
h1 = var(r);
fu = @(X)fscr_igarch(X, r, h1);

% Symbolic variables
a = sym('a', [1, 2], 'real');
b = sym('b', [1, 2], 'real');

% Kernels
k_1 = (1e-5 + (a - b) * (a - b)') .^ (-0.1);
k_2 = (2e-5 + log(1 + (a - b) * (a - b)')) .^ (-1);

% Optimisers
lb = [0.002, 0.05];
ub = [0.04, 0.2];
mu0 = (lb + ub) ./ 2;
Sigma0 = diag([1e-4, 1e-3]);
nStep = 100;
fmin_1 = @(f, X)fmin_adares(f, X, 3, mu0, Sigma0, 1e-5, 20, lb, ub, [], []);
fmin_2 = @(f, X)fmin_adamc(f, X, 20, mu0, Sigma0, 1e-5, 20, lb, ub, [], []);
fmin_3 = @(f, X)fmin_gsrtn(f, X, lb, ub, [nStep, nStep]);
fmin_4 = @(f, X)fmin_gs2d(f, lb, ub, [nStep, nStep]);
fmin_5 = @(f, X)fmin_ps(f, 2, 10, lb, ub);

% Run various greedy KSD algorithms
[X_1, ~, nEval_1] = stein_greedy(2, fu, k_1, fmin_1, nPart, []);
[X_2, ~, nEval_2] = stein_greedy(2, fu, k_2, fmin_1, nPart, []);
[X_3, ~, nEval_3] = stein_greedy(2, fu, k_1, fmin_2, nPart, []);
[X_4, ~, nEval_4] = stein_greedy(2, fu, k_2, fmin_2, nPart, []);

% Run MCMC
nMcmc = 100000;
nBurn = 5000;
Chain = mcmc_igarch(r, nMcmc + nBurn, nBurn);

% Compute EMD
nIid = 300;
nRep = 5;
Emd_1 = zeros(nPart, nRep);
Emd_2 = zeros(nPart, nRep);
Emd_3 = zeros(nPart, nRep);
Emd_4 = zeros(nPart, nRep);
alg = {'1', '2', '3', '4'};
SegY = [ones(nPart, 1), [1:nPart]'];
for i = 1:numel(alg)
    YPts = eval(['X_', alg{i}]);
    for j = 1:nRep
        XPts = Chain(randperm(nMcmc, nIid), :);
        save('pts.mat', 'XPts', 'YPts', 'SegY');
        system('julia l1emd.jl pts.mat emd.mat');
        load('emd.mat');
        eval(['Emd_', alg{i}, '(:,', num2str(j), ')=emd;']);
    end
end

% Generate plots
mkr = {'x', 'd', 'o', '+'};
for i = 1:numel(mkr)
    xx = log(cumsum(eval(['nEval_', alg{i}])));
    yy = log(mean(eval(['Emd_', alg{i}]), 2));
    plot(xx, yy, mkr{i}, 'MarkerSize', 6);
    hold on;
end
axis([0, 13, -6.5, -2.5]);
set(gca, 'fontsize', 19);
xlabel('log n_{eval}', 'fontsize', 19);
ylabel('log W_{P}', 'fontsize', 19);
legend({ ...
    'NM k1', ...
    'NM k2', ...
    'MC k1', ...
    'MC k2'}, ...
    'location', 'northeast', ...
    'fontsize', 16);

% Window setting
set(gcf, 'renderer', 'painters');
set(gcf, 'units', 'centimeters');
set(gcf, 'position', [0.5, 1.5, 15, 12]);

% Print setting
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'paperpositionmode', 'manual');
set(gcf, 'paperposition', [0, 0, 15, 12]);
set(gcf, 'papertype', '<custom>');
set(gcf, 'papersize', [15, 12]);

% Print to PDF
name = sprintf('genemd_igarch_stein_greedy_%d', nPart);
print(name, '-dpdf');

% Save output
save(strcat(name, '.mat'));
