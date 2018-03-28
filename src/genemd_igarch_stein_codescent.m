%%
% File: genemd_igarch_stein_codescent.m
% Purpose:
% Generates EMD plots for various KSD coordinate-descent algorithms. The
% target is the posterior of an IGARCH model.
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
k_1 = (1e-5 + (a - b) * (a - b)') .^ (-0.5);
k_2 = (8e-5 + log(1 + (a - b) * (a - b)')) .^ (-1);

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

% Initial particles
X0_1 = [unifrnd(lb(1), ub(1), nPart, 1), unifrnd(lb(2), ub(2), nPart, 1)];
X0_2 = mvnrnd(mu0, Sigma0, nPart);
X0_3 = circ(mu0, 0.001, nPart);
X0_4 = bx([0, 0], 0.12, nPart) * diag([0.3, 1]) + repmat(mu0, nPart, 1);
X0 = X0_4;

% Run various coordinate-descent KSD algorithms
nIter = 6 .* nPart;
[X_1, nEval_1] = stein_codescent(fu, k_1, fmin_1, X0, nIter, []);
[X_2, nEval_2] = stein_codescent(fu, k_2, fmin_1, X0, nIter, []);
[X_3, nEval_3] = stein_codescent(fu, k_1, fmin_2, X0, nIter, []);
[X_4, nEval_4] = stein_codescent(fu, k_2, fmin_2, X0, nIter, []);

% Run MCMC
nMcmc = 100000;
nBurn = 5000;
Chain = mcmc_igarch(r, nMcmc + nBurn, nBurn);

% Compute EMD
nIid = 300;
nRep = 5;
Emd_1 = zeros(nIter, nRep);
Emd_2 = zeros(nIter, nRep);
Emd_3 = zeros(nIter, nRep);
Emd_4 = zeros(nIter, nRep);
alg = {'1', '2', '3', '4'};
segEnd = nPart:nPart:(nPart .* nIter);
segBeg = segEnd - nPart + 1;
SegY = [segBeg', segEnd'];
for i = 1:numel(alg)
    YPts = eval(['X_', alg{i}]);
    YPts = reshape(permute(YPts, [1, 3, 2]), [], 2, 1);
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
    'location', 'southwest', ...
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
name = sprintf('genemd_igarch_stein_codescent_%d', nPart);
print(name, '-dpdf');

% Save output
save(strcat(name, '.mat'));
