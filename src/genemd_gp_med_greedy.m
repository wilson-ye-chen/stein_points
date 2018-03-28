%%
% File: genemd_gp_med_greedy.m
% Purpose:
% Generates EMD plots for various MED greedy algorithms. The target is the
% posterior of a Gaussian process regression model.
% Date: February 8, 2018
%%

% Number of particles
nPart = 100;

% Load the LIDAR data.
load('data_lidar.mat');
xVar = range;
yVar = logratio;

% Target score and density functions
sigma = 0.2;
fu = @(Theta)fscr_gp(Theta, yVar, xVar, sigma, false);
[~, fp] = fu([0, 0]);

% Symbolic variables
a = sym('a', [1, 2], 'real');
b = sym('b', [1, 2], 'real');

% Tuning parameter
k_1 = 1;

% Optimisers
lb = [-5, -13];
ub = [5, -7];
mu0 = [0, -10];
Sigma0 = diag([25, 25]);
nStep = 100;
fmin_1 = @(f, X)fmin_adares(f, X, 3, mu0, Sigma0, 1, 20, lb, ub, [], []);
fmin_2 = @(f, X)fmin_adamc(f, X, 20, mu0, Sigma0, 1, 20, lb, ub, [], []);
fmin_3 = @(f, X)fmin_gsrtn(f, X, lb, ub, [nStep, nStep]);
fmin_4 = @(f, X)fmin_gs2d(f, lb, ub, [nStep, nStep]);
fmin_5 = @(f, X)fmin_ps(f, 2, 10, lb, ub);

% Run various greedy MED algorithms
[X_1, ~, nEval_1] = med_greedy(2, fp, fmin_1, k_1, nPart);
[X_2, ~, nEval_2] = med_greedy(2, fp, fmin_2, k_1, nPart);

% Run MCMC
nMcmc = 100000;
nBurn = 5000;
Chain = mcmc_gp(yVar, xVar, sigma, nMcmc + nBurn, nBurn);

% Compute EMD
nIid = 300;
nRep = 5;
Emd_1 = zeros(nPart, nRep);
Emd_2 = zeros(nPart, nRep);
alg = {'1', '2'};
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
mkr = {'x', 'd'};
for i = 1:numel(mkr)
    xx = log(cumsum(eval(['nEval_', alg{i}])));
    yy = log(mean(eval(['Emd_', alg{i}]), 2));
    plot(xx, yy, mkr{i}, 'MarkerSize', 6);
    hold on;
end
axis([0, 12, -1.65, 1.8]);
set(gca, 'fontsize', 19);
xlabel('log n_{eval}', 'fontsize', 19);
ylabel('log W_{P}', 'fontsize', 19);
legend({ ...
    'NM \delta=1', ...
    'MC \delta=1'}, ...
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
name = sprintf('genemd_gp_med_greedy_%d', nPart);
print(name, '-dpdf');

% Save output
save(strcat(name, '.mat'));
