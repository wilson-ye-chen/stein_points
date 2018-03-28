%%
% File: genemd_igarch_svgd.m
% Purpose:
% Generates EMD plots for various SVGD algorithms. The target is the
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
k_1 = (5e-6 + (a - b) * (a - b)') .^ (-0.3);
k_2 = (1e-5 + log(1 + (a - b) * (a - b)')) .^ (-1);

% Initial particles
lb = [0.002, 0.05];
ub = [0.04, 0.2];
mu0 = (lb + ub) ./ 2;
Sigma0 = diag([1e-4, 1e-3]);
X0_1 = [unifrnd(lb(1), ub(1), nPart, 1), unifrnd(lb(2), ub(2), nPart, 1)];
X0_2 = mvnrnd(mu0, Sigma0, nPart);
X0_3 = circ(mu0, 0.001, nPart);
X0_4 = bx([0, 0], 0.12, nPart) * diag([0.3, 1]) + repmat(mu0, nPart, 1);
X0 = X0_4;

% Run various SVGD algorithms
nIter = 200;
fstp = @(i, Phi)fstp_adagrad(i, Phi, 1e-3, 0.9);
[X_1, nEval_1] = mysvgd(fu, k_1, X0, fstp, nIter);
[X_2, nEval_2] = mysvgd(fu, k_2, X0, fstp, nIter);

% Run MCMC
nMcmc = 100000;
nBurn = 5000;
Chain = mcmc_igarch(r, nMcmc + nBurn, nBurn);

% Compute EMD
nIid = 300;
nRep = 5;
Emd_1 = zeros(nIter, nRep);
Emd_2 = zeros(nIter, nRep);
alg = {'1', '2'};
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
mkr = {'x', 'd'};
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
    'k1', ...
    'k2'}, ...
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
name = sprintf('genemd_igarch_svgd_%d', nPart);
print(name, '-dpdf');

% Save output
save(strcat(name, '.mat'));
