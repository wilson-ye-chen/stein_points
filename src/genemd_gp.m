%%
% File: genemd_gp.m
% Purpose:
% Generates EMD plots for various discrete approximation algorithms. The
% target is the posterior of a Gaussian process regression model.
% Date: February 4, 2018
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

% Initial particles
X0_1 = [unifrnd(-5, 5, nPart, 1), unifrnd(-13, -7, nPart, 1)];
X0_2 = mvnrnd([0, -10], eye(2), nPart);
X0_3 = circ([0, -10], 0.1, nPart);
X0_4 = bx([0, -10], 10, nPart);

% (a) Monte Carlo
X_1 = mcmc_gp(yVar, xVar, sigma, nPart, 0);
nEval_1 = [0; ones(nPart - 1, 1)];
Seg_1 = [ones(nPart, 1), [1:nPart]'];

% (b) Stein Greedy
k = (0.5 + log(1 + (a - b) * (a - b)')) .^ (-1);
[X_2, ~, nEval_2] = stein_greedy(2, fu, k, fmin_2, nPart, []);
Seg_2 = [ones(nPart, 1), [1:nPart]'];

% (c) Stein Herding
k = (0.1 + log(1 + (a - b) * (a - b)')) .^ (-1);
[X_3, ~, nEval_3] = stein_fw(2, fu, k, fmin_2, nPart, []);
Seg_3 = [ones(nPart, 1), [1:nPart]'];

% (d) Stein Co-Descent
nIter_4 = 6 .* nPart;
k = (0.5 + log(1 + (a - b) * (a - b)')) .^ (-1);
[X_4, nEval_4] = stein_codescent(fu, k, fmin_2, X0_4, nIter_4, []);
X_4 = reshape(permute(X_4, [1, 3, 2]), [], 2, 1);
segEnd = nPart:nPart:(nPart .* nIter_4);
segBeg = segEnd - nPart + 1;
Seg_4 = [segBeg', segEnd'];

% (e) MED is numerically unstable for the GP example.

% (f) SVGD
nIter_5 = 200;
fstp = @(i, Phi)fstp_adagrad(i, Phi, 0.1, 0.9);
k = (0.1 + (a - b) * (a - b)') .^ (-0.9);
[X_5, nEval_5] = mysvgd(fu, k, X0_4, fstp, nIter_5);
X_5 = reshape(permute(X_5, [1, 3, 2]), [], 2, 1);
segEnd = nPart:nPart:(nPart .* nIter_5);
segBeg = segEnd - nPart + 1;
Seg_5 = [segBeg', segEnd'];

% Run MCMC
nMcmc = 100000;
nBurn = 5000;
Chain = mcmc_gp(yVar, xVar, sigma, nMcmc + nBurn, nBurn);

% Compute EMD
nIid = 300;
nRep = 5;
Emd_1 = zeros(nPart, nRep);
Emd_2 = zeros(nPart, nRep);
Emd_3 = zeros(nPart, nRep);
Emd_4 = zeros(nIter_4, nRep);
Emd_5 = zeros(nIter_5, nRep);
alg = {'1', '2', '3', '4', '5'};
for i = 1:numel(alg)
    YPts = eval(['X_', alg{i}]);
    SegY = eval(['Seg_', alg{i}]);
    for j = 1:nRep
        XPts = Chain(randperm(nMcmc, nIid), :);
        save('pts.mat', 'XPts', 'YPts', 'SegY');
        system('julia l1emd.jl pts.mat emd.mat');
        load('emd.mat');
        eval(['Emd_', alg{i}, '(:,', num2str(j), ')=emd;']);
    end
end

% Generate plots
mkr = {'x', 'd', 'v', 'o', 's'};
for i = 1:numel(mkr)
    xx = log(cumsum(eval(['nEval_', alg{i}])));
    yy = log(mean(eval(['Emd_', alg{i}]), 2));
    plot(xx, yy, mkr{i}, 'MarkerSize', 6);
    hold on;
end
set(gca, 'fontsize', 16);
xlabel('log n_{eval}', 'fontsize', 16);
ylabel('log W_{P}', 'fontsize', 16);
legend({ ...
    'MCMC', ...
    'Stein Greedy', ...
    'Stein Herding', ...
    'Stein Co-Descent', ...
    'SVGD'}, ...
    'location', 'southwest', ...
    'fontsize', 13);

% Window setting
set(gcf, 'renderer', 'painters');
set(gcf, 'units', 'centimeters');
set(gcf, 'position', [0.5, 1.5, 20, 12]);

% Print setting
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'paperpositionmode', 'manual');
set(gcf, 'paperposition', [0, 0, 20, 12]);
set(gcf, 'papertype', '<custom>');
set(gcf, 'papersize', [20, 12]);

% Print to PDF
name = sprintf('genemd_gp_%d', nPart);
print(name, '-dpdf');

% Save output
save(strcat(name, '.mat'));
