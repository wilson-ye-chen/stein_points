%%
% File: genemd_gp_svgd.m
% Purpose:
% Generates EMD plots for various SVGD algorithms. The target is the
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
[~, post] = fu([0, 0]);
fp = @(theta)log(post([theta(1), theta(2)]));

% Symbolic variables
a = sym('a', [1, 2], 'real');
b = sym('b', [1, 2], 'real');

% Kernels
k_1 = (0.1 + (a - b) * (a - b)') .^ (-0.9);
k_2 = (0.5 + log(1 + (a - b) * (a - b)')) .^ (-1);

% Initial particles
X0_1 = [unifrnd(-5, 5, nPart, 1), unifrnd(-13, -7, nPart, 1)];
X0_2 = mvnrnd([0, -10], eye(2), nPart);
X0_3 = circ([0, -10], 0.1, nPart);
X0_4 = bx([0, -10], 10, nPart);
X0 = X0_4;

% Run various SVGD algorithms
nIter = 200;
fstp = @(i, Phi)fstp_adagrad(i, Phi, 0.1, 0.9);
[X_1, nEval_1] = mysvgd(fu, k_1, X0, fstp, nIter);
[X_2, nEval_2] = mysvgd(fu, k_2, X0, fstp, nIter);

% Run MCMC
nMcmc = 100000;
nBurn = 5000;
Chain = mcmc_gp(yVar, xVar, sigma, nMcmc + nBurn, nBurn);

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
axis([0, 12, -1.65, 1.8]);
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
name = sprintf('genemd_gp_svgd_%d', nPart);
print(name, '-dpdf');

% Save output
save(strcat(name, '.mat'));
