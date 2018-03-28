%%
% File: selkrn_gp_stein_codescent.m
% Purpose:
% Plots EMD values for KSD coordinate-descent algorithms with various kernels.
% The target is the posterior of a Gaussian process regression model.
% Date: February 7, 2018
%%

% Number of particles and iterations
nPart = 100;
nIter = 6 .* nPart;

% Load the LIDAR data.
load('data_lidar.mat');
xVar = range;
yVar = logratio;

% Target score and density functions
sigma = 0.2;
fu = @(Theta)fscr_gp(Theta, yVar, xVar, sigma, false);
[~, post] = fu([0, 0]);
fp = @(theta)log(post([theta(1), theta(2)]));

% Symbolic variables
a = sym('a', [1, 2], 'real');
b = sym('b', [1, 2], 'real');

% Kernel parameters
alpha = [0.1, 0.5, 1, 2, 4, 8];
beta = [-0.1, -0.3, -0.5, -0.7, -0.9];
nAlpha = numel(alpha);
nBeta = numel(beta);

% Optimisers
lb = [-5, -13];
ub = [5, -7];
mu0 = [0, -10];
Sigma0 = diag([25, 25]);
nStep = 100;
fmin_1 = @(f, X)fmin_gs2d(f, lb, ub, [nStep, nStep]);
fmin_2 = @(f, X)fmin_adamc(f, X, 20, mu0, Sigma0, 1, 20, lb, ub, [], []);
fmin_3 = @(f, X)fmin_adares(f, X, 3, mu0, Sigma0, 1, 20, lb, ub, [], []);
fmin_4 = @(f, X)fmin_ps(f, 2, 10, lb, ub);
fmin = fmin_3;

% Initial particles
X0_1 = [unifrnd(-5, 5, nPart, 1), unifrnd(-13, -7, nPart, 1)];
X0_2 = mvnrnd([0, -10], eye(2), nPart);
X0_3 = circ([0, -10], 0.1, nPart);
X0_4 = bx([0, -10], 10, nPart);
X0 = X0_4;

% IMQ kernel
nPerm_1 = nAlpha .* nBeta;
Param_1 = zeros(nPerm_1, 2);
X_1 = zeros(nPart, 2, nPerm_1);
t = 1;
for i = 1:nAlpha
    for j = 1:nBeta
        k = (alpha(i) + (a - b) * (a - b)') .^ beta(j);
        X = stein_codescent(fu, k, fmin, X0, nIter, []);
        X_1(:, :, t) = X(:, :, end);
        Param_1(t, :) = [alpha(i), beta(j)];
        t = t + 1;
    end
end

% The 'log' kernel
nPerm_2 = nAlpha;
Param_2 = zeros(nPerm_2, 1);
X_2 = zeros(nPart, 2, nPerm_2);
for i = 1:nAlpha
    k = (alpha(i) + log(1 + (a - b) * (a - b)')) .^ (-1);
    X = stein_codescent(fu, k, fmin, X0, nIter, []);
    X_2(:, :, i) = X(:, :, end);
    Param_2(i) = alpha(i);
end

% Flatten
Y_1 = reshape(permute(X_1, [1, 3, 2]), [], 2, 1);
Y_2 = reshape(permute(X_2, [1, 3, 2]), [], 2, 1);

% Run MCMC
nMcmc = 100000;
nBurn = 5000;
Chain = mcmc_gp(yVar, xVar, sigma, nMcmc + nBurn, nBurn);

% Compute EMD
nIid = 300;
nRep = 5;
Emd_1 = zeros(nPerm_1, nRep);
Emd_2 = zeros(nPerm_2, nRep);
krn = {'1', '2'};
for i = 1:numel(krn)
    YPts = eval(['Y_', krn{i}]);
    segEnd = nPart:nPart:size(YPts, 1);
    segBeg = segEnd - nPart + 1;
    SegY = [segBeg', segEnd'];
    for j = 1:nRep
        XPts = Chain(randperm(nMcmc, nIid), :);
        save('pts.mat', 'XPts', 'YPts', 'SegY');
        system('julia l1emd.jl pts.mat emd.mat');
        load('emd.mat');
        eval(['Emd_', krn{i}, '(:,', num2str(j), ')=emd;']);
    end
end

% Find best parameters
logMeanEmd_1 = log(mean(Emd_1, 2));
logMeanEmd_2 = log(mean(Emd_2, 2));
[emd_1, i_1] = min(logMeanEmd_1);
[emd_2, i_2] = min(logMeanEmd_2);
par_1 = Param_1(i_1, :);
par_2 = Param_2(i_2);

% Generate plots
hold on;
plot(1:nPerm_1, logMeanEmd_1, '.-', 'markersize', 13, 'linewidth', 1);
plot(1:nPerm_2, logMeanEmd_2, '.-', 'markersize', 13, 'linewidth', 1);
plot(i_1, emd_1, 'ko', 'markersize', 8, 'linewidth', 1);
plot(i_2, emd_2, 'ko', 'markersize', 8, 'linewidth', 1);
set(gca, 'fontsize', 16);
xlabel('Permutation Index', 'fontsize', 16);
ylabel('log W_P','fontsize', 16);
h = legend({ ...
    sprintf('k1: \\alpha=%.1f, \\beta=%.1f', par_1(1), par_1(2)), ...
    sprintf('k2: \\alpha=%.1f', par_2)}, ...
    'fontsize', 13, ...
    'location', 'best');

% Window setting
set(gcf, 'renderer', 'painters');
set(gcf, 'units', 'centimeters');
set(gcf, 'position', [0.5, 1.5, 16, 10]);

% Print setting
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'paperpositionmode', 'manual');
set(gcf, 'paperposition', [0, 0, 16, 10]);
set(gcf, 'papertype', '<custom>');
set(gcf, 'papersize', [16, 10]);

% Print to PDF
name = sprintf('selkrn_gp_stein_codescent_%d', nPart);
print(name, '-dpdf');

% Save output
save(strcat(name, '.mat'));
