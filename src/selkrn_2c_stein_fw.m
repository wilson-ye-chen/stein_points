%%
% File: selkrn_2c_stein_fw.m
% Purpose:
% Plots EMD values for KSD Frank-Wolfe algorithms with various kernels. The
% target is a two component Gaussian mixture distribution.
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

% Kernel parameters
alpha = [0.1, 0.5, 1, 2, 4, 8];
beta = [-0.1, -0.3, -0.5, -0.7, -0.9];
nAlpha = numel(alpha);
nBeta = numel(beta);

% Optimisers
lb = [-5, -5];
ub = [5, 5];
mu0 = [0, 0];
Sigma0 = diag([25, 25]);
nStep = 100;
fmin_1 = @(f, X)fmin_gs2d(f, lb, ub, [nStep, nStep]);
fmin_2 = @(f, X)fmin_adamc(f, X, 20, mu0, Sigma0, 1, 20, lb, ub, [], []);
fmin_3 = @(f, X)fmin_adares(f, X, 3, mu0, Sigma0, 1, 20, lb, ub, [], []);
fmin_4 = @(f, X)fmin_ps(f, 2, 10, lb, ub);
fmin = fmin_3;

% IMQ kernel
nPerm_1 = nAlpha .* nBeta;
Param_1 = zeros(nPerm_1, 2);
X_1 = zeros(nPart, 2, nPerm_1);
t = 1;
for i = 1:nAlpha
    for j = 1:nBeta
        k = (alpha(i) + (a - b) * (a - b)') .^ beta(j);
        X_1(:, :, t) = stein_fw(2, fu, k, fmin, nPart, []);
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
    X_2(:, :, i) = stein_fw(2, fu, k, fmin, nPart, []);
    Param_2(i) = alpha(i);
end

% The score-based kernel
nPerm_3 = nAlpha .* nBeta;
Param_3 = zeros(nPerm_3, 2);
X_3 = zeros(nPart, 2, nPerm_3);
t = 1;
for i = 1:nAlpha
    for j = 1:nBeta
        k = symdef_k2c(alpha(i), beta(j));
        X_3(:, :, t) = stein_fw(2, fu, k, fmin, nPart, []);
        Param_3(t, :) = [alpha(i), beta(j)];
        t = t + 1;
    end
end

% Flatten
Y_1 = reshape(permute(X_1, [1, 3, 2]), [], 2, 1);
Y_2 = reshape(permute(X_2, [1, 3, 2]), [], 2, 1);
Y_3 = reshape(permute(X_3, [1, 3, 2]), [], 2, 1);

% Compute EMD
nIid = 300;
nRep = 5;
Emd_1 = zeros(nPerm_1, nRep);
Emd_2 = zeros(nPerm_2, nRep);
Emd_3 = zeros(nPerm_3, nRep);
krn = {'1', '2', '3'};
for i = 1:numel(krn)
    YPts = eval(['Y_', krn{i}]);
    segEnd = nPart:nPart:size(YPts, 1);
    segBeg = segEnd - nPart + 1;
    SegY = [segBeg', segEnd'];
    for j = 1:nRep
        XPts = gaussmix_rnd(Mu, C, w, nIid);
        save('pts.mat', 'XPts', 'YPts', 'SegY');
        system('julia l1emd.jl pts.mat emd.mat');
        load('emd.mat');
        eval(['Emd_', krn{i}, '(:,', num2str(j), ')=emd;']);
    end
end

% Find best parameters
logMeanEmd_1 = log(mean(Emd_1, 2));
logMeanEmd_2 = log(mean(Emd_2, 2));
logMeanEmd_3 = log(mean(Emd_3, 2));
[emd_1, i_1] = min(logMeanEmd_1);
[emd_2, i_2] = min(logMeanEmd_2);
[emd_3, i_3] = min(logMeanEmd_3);
par_1 = Param_1(i_1, :);
par_2 = Param_2(i_2);
par_3 = Param_3(i_3, :);

% Generate plots
hold on;
plot(1:nPerm_1, logMeanEmd_1, '.-', 'markersize', 13, 'linewidth', 1);
plot(1:nPerm_2, logMeanEmd_2, '.-', 'markersize', 13, 'linewidth', 1);
plot(1:nPerm_3, logMeanEmd_3, '.-', 'markersize', 13, 'linewidth', 1);
plot(i_1, emd_1, 'ko', 'markersize', 8, 'linewidth', 1);
plot(i_2, emd_2, 'ko', 'markersize', 8, 'linewidth', 1);
plot(i_3, emd_3, 'ko', 'markersize', 8, 'linewidth', 1);
set(gca, 'fontsize', 16);
xlabel('Permutation Index', 'fontsize', 16);
ylabel('log W_{P}', 'fontsize', 16);
legend({ ...
    sprintf('k1: \\alpha=%.1f, \\beta=%.1f', par_1(1), par_1(2)), ...
    sprintf('k2: \\alpha=%.1f', par_2), ...
    sprintf('k3: \\alpha=%.1f, \\beta=%.1f', par_3(1), par_3(2))}, ...
    'location', 'best', ...
    'fontsize', 13);

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
name = sprintf('selkrn_2c_stein_fw_%d', nPart);
print(name, '-dpdf');

% Save output
save(strcat(name, '.mat'));
