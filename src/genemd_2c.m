%%
% File: genemd_2c.m
% Purpose:
% Generates EMD plots for various discrete approximation algorithms. The
% target is a two-component Gaussian mixture distribution.
% Date: February 9, 2018
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

% (a) Monte Carlo
X_1 = gaussmix_rnd(Mu, C, w, nPart);
nEval_1 = ones(nPart, 1);
Seg_1 = [ones(nPart, 1), [1:nPart]'];

% (b) Stein Greedy
k = (4 + log(1 + (a - b) * (a - b)')) .^ (-1);
[X_2, ~, nEval_2] = stein_greedy(2, fu, k, fmin_2, nPart, []);
Seg_2 = [ones(nPart, 1), [1:nPart]'];

% (c) Stein Herding
k = (0.5 + (a - b) * (a - b)') .^ (-0.7);
[X_3, ~, nEval_3] = stein_fw(2, fu, k, fmin_2, nPart, []);
Seg_3 = [ones(nPart, 1), [1:nPart]'];

% (d) Stein Co-Descent
nIter_4 = 6 .* nPart;
k = (2 + log(1 + (a - b) * (a - b)')) .^ (-1);
[X_4, nEval_4] = stein_codescent(fu, k, fmin_2, X0_3, nIter_4, []);
X_4 = reshape(permute(X_4, [1, 3, 2]), [], 2, 1);
segEnd = nPart:nPart:(nPart .* nIter_4);
segBeg = segEnd - nPart + 1;
Seg_4 = [segBeg', segEnd'];

% (f) MED
[X_5, ~, nEval_5] = med_greedy(2, fp, fmin_2, 4, nPart);
Seg_5 = [ones(nPart, 1), [1:nPart]'];

% (f) SVGD
nIter_6 = 200;
fstp = @(i, Phi)fstp_adagrad(i, Phi, 0.1, 0.9);
k = (0.5 + (a - b) * (a - b)') .^ (-0.9);
[X_6, nEval_6] = mysvgd(fu, k, X0_3, fstp, nIter_6);
X_6 = reshape(permute(X_6, [1, 3, 2]), [], 2, 1);
segEnd = nPart:nPart:(nPart .* nIter_6);
segBeg = segEnd - nPart + 1;
Seg_6 = [segBeg', segEnd'];

% Compute EMD
nIid = 300;
nRep = 5;
Emd_1 = zeros(nPart, nRep);
Emd_2 = zeros(nPart, nRep);
Emd_3 = zeros(nPart, nRep);
Emd_4 = zeros(nIter_4, nRep);
Emd_5 = zeros(nPart, nRep);
Emd_6 = zeros(nIter_6, nRep);
alg = {'1', '2', '3', '4', '5', '6'};
for i = 1:numel(alg)
    YPts = eval(['X_', alg{i}]);
    SegY = eval(['Seg_', alg{i}]);
    for j = 1:nRep
        XPts = gaussmix_rnd(Mu, C, w, nIid);
        save('pts.mat', 'XPts', 'YPts', 'SegY');
        system('julia l1emd.jl pts.mat emd.mat');
        load('emd.mat');
        eval(['Emd_', alg{i}, '(:,', num2str(j), ')=emd;']);
    end
end

% Generate plots
mkr = {'x', 'd', 'v', 'o', '^', 's'};
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
    'Monte Carlo', ...
    'Stein Greedy', ...
    'Stein Herding', ...
    'Stein Co-Descent', ...
    'MED', ...
    'SVGD'}, ...
    'position', [0.265, 0.31, 0, 0], ...
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
name = sprintf('genemd_2c_%d', nPart);
print(name, '-dpdf');

% Save output
save(strcat(name, '.mat'));
