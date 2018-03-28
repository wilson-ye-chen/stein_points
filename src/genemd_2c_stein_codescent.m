%%
% File: genemd_2c_stein_codescent.m
% Purpose:
% Generates EMD plots for various KSD coordinate descent algorithms. The
% target is a two-component Gaussian mixture distribution.
% Date: February 8, 2018
%%

% Number of particles
nPart = 100;

% Target density and score functions
[Mu, C, w] = gmparam_2c();
fp = @(X)fp_gaussmix(X, Mu, C, w);
fu = @(X)fscr_gaussmix(X, Mu, C, w);

% Symbolic variables
a = sym('a', [1, 2], 'real');
b = sym('b', [1, 2], 'real');

% Kernels
k_1 = (0.5 + (a - b) * (a - b)') .^ (-0.3);
k_2 = (2 + log(1 + (a - b) * (a - b)')) .^ (-1);
k_3 = symdef_k2c(8, -0.9);

% Optimisers
lb = [-5, -5];
ub = [5, 5];
mu0 = [0, 0];
Sigma0 = diag([25, 25]);
nStep = 100;
fmin_1 = @(f, X)fmin_adares(f, X, 3, mu0, Sigma0, 1, 20, lb, ub, [], []);
fmin_2 = @(f, X)fmin_adamc(f, X, 20, mu0, Sigma0, 1, 20, lb, ub, [], []);
fmin_3 = @(f, X)fmin_gsrtn(f, X, lb, ub, [nStep, nStep]);
fmin_4 = @(f, X)fmin_gs2d(f, lb, ub, [nStep, nStep]);
fmin_5 = @(f, X)fmin_ps(f, 2, 10, lb, ub);

% Initial particles
X0_1 = unifrnd(-5, 5, nPart, 2);
X0_2 = circ([0, 0], 0.1, nPart);
X0_3 = bx([0, 0], 10, nPart);

% Run various KSD coordinate descent algorithms
nIter = 6 .* nPart;
[X_1, nEval_1] = stein_codescent(fu, k_1, fmin_1, X0_3, nIter, []);
[X_2, nEval_2] = stein_codescent(fu, k_2, fmin_1, X0_3, nIter, []);
[X_3, nEval_3] = stein_codescent(fu, k_3, fmin_1, X0_3, nIter, []);
[X_4, nEval_4] = stein_codescent(fu, k_1, fmin_2, X0_3, nIter, []);
[X_5, nEval_5] = stein_codescent(fu, k_2, fmin_2, X0_3, nIter, []);
[X_6, nEval_6] = stein_codescent(fu, k_3, fmin_2, X0_3, nIter, []);
[X_7, nEval_7] = stein_codescent(fu, k_1, fmin_3, X0_3, nIter, []);
[X_8, nEval_8] = stein_codescent(fu, k_2, fmin_3, X0_3, nIter, []);
[X_9, nEval_9] = stein_codescent(fu, k_3, fmin_3, X0_3, nIter, []);

% Compute EMD
nIid = 300;
nRep = 5;
Emd_1 = zeros(nIter, nRep);
Emd_2 = zeros(nIter, nRep);
Emd_3 = zeros(nIter, nRep);
Emd_4 = zeros(nIter, nRep);
Emd_5 = zeros(nIter, nRep);
Emd_6 = zeros(nIter, nRep);
Emd_7 = zeros(nIter, nRep);
Emd_8 = zeros(nIter, nRep);
Emd_9 = zeros(nIter, nRep);
alg = {'1', '2', '3', '4', '5', '6', '7', '8', '9'};
segEnd = nPart:nPart:(nPart .* nIter);
segBeg = segEnd - nPart + 1;
SegY = [segBeg', segEnd'];
for i = 1:numel(alg)
    YPts = eval(['X_', alg{i}]);
    YPts = reshape(permute(YPts, [1, 3, 2]), [], 2, 1);
    for j = 1:nRep
        XPts = gaussmix_rnd(Mu, C, w, nIid);
        save('pts.mat', 'XPts', 'YPts', 'SegY');
        system('julia l1emd.jl pts.mat emd.mat');
        load('emd.mat');
        eval(['Emd_', alg{i}, '(:,', num2str(j), ')=emd;']);
    end
end

% Generate plots
mkr = {'x', 'd', 'v', 'o', '+', '^', 's', '*', '>'};
for i = 1:numel(mkr)
    xx = log(cumsum(eval(['nEval_', alg{i}])));
    yy = log(mean(eval(['Emd_', alg{i}]), 2));
    plot(xx, yy, mkr{i}, 'MarkerSize', 6);
    hold on;
end
axis([0, 16, -1.1, 1.8]);
set(gca, 'fontsize', 19);
xlabel('log n_{eval}', 'fontsize', 19);
ylabel('log W_{P}', 'fontsize', 19);
legend({ ...
    'NM k1', ...
    'NM k2', ...
    'NM k3', ...
    'MC k1', ...
    'MC k2', ...
    'MC k3', ...
    'GS k1', ...
    'GS k2', ...
    'GS k3'}, ...
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
name = sprintf('genemd_2c_stein_codescent_%d', nPart);
print(name, '-dpdf');

% Save output
save(strcat(name, '.mat'));
