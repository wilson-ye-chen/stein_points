%%
% File: genemd_gp_mc.m
% Purpose:
% Generates EMD plots for the Monte Carlo algorithm. The target is the
% posterior of a Gaussian process regression model.
% Date: February 8, 2018
%%

% Number of particles
nPart = 100;

% Load the LIDAR data.
load('data_lidar.mat');
xVar = range;
yVar = logratio;
sigma = 0.2;

% Run Metropolis
X_1 = mcmc_gp(yVar, xVar, sigma, nPart, 0);
nEval_1 = [0; ones(nPart - 1, 1)];

% Run Metropolis with thinning
nThin = 100;
nChain = nPart .* nThin;
X = mcmc_gp(yVar, xVar, sigma, nChain, 0);
X_2 = X(1:nThin:nChain, :);
nEval_2 = [0; repmat(nThin, nPart - 1, 1)];

% Run MCMC to generate independent samples
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
    'MCMC', ...
    'MCMC-Thin'}, ...
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
name = sprintf('genemd_gp_mc_%d', nPart);
print(name, '-dpdf');

% Save output
save(strcat(name, '.mat'));
