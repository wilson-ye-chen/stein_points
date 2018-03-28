%%
% File: genemd_2c_svgd.m
% Purpose:
% Generates EMD plots for various SVGD algorithms. The target is a two-
% component Gaussian mixture distribution.
% Date: February 7, 2018
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
k_1 = (0.5 + (a - b) * (a - b)') .^ (-0.9);
k_2 = (2 + log(1 + (a - b) * (a - b)')) .^ (-1);
k_3 = symdef_k2c(0.1, -0.7);

% Initial particles
X0_1 = unifrnd(-5, 5, nPart, 2);
X0_2 = circ([0, 0], 0.1, nPart);
X0_3 = bx([0, 0], 10, nPart);

% Run various SVGD algorithms
nIter = 200;
fstp = @(i, Phi)fstp_adagrad(i, Phi, 0.1, 0.9);
[X_1, nEval_1] = mysvgd(fu, k_1, X0_3, fstp, nIter);
[X_2, nEval_2] = mysvgd(fu, k_2, X0_3, fstp, nIter);
[X_3, nEval_3] = mysvgd(fu, k_3, X0_3, fstp, nIter);

% Compute EMD
nIid = 300;
nRep = 5;
Emd_1 = zeros(nIter, nRep);
Emd_2 = zeros(nIter, nRep);
Emd_3 = zeros(nIter, nRep);
alg = {'1', '2', '3'};
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
mkr = {'x', 'd', 'v'};
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
    'k1', ...
    'k2', ...
    'k3'}, ...
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
name = sprintf('genemd_2c_svgd_%d', nPart);
print(name, '-dpdf');

% Save output
save(strcat(name, '.mat'));
