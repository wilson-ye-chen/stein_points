%%
% File: stein_fw_test.m
% Purpose:
% Runs the Frank-Wolfe Stein sampling algorithm on various targets.
% Date: January 22, 2018
%%

% Test case
testCase = 1;

% Algorithm configurations
smplr = @stein_fw;
nIter = 100;

% Optimisers
lb = [-10, -10];
ub = [10, 10];
mu0 = [0, 0];
Sigma0 = diag([25, 25]);
nStep = 100;
fmin_1 = @(f, X)fmin_gs2d(f, lb, ub, [nStep, nStep]);
fmin_2 = @(f, X)fmin_adamc(f, X, 20, mu0, Sigma0, 1, 20, lb, ub, [], []);
fmin_3 = @(f, X)fmin_adares(f, X, 3, mu0, Sigma0, 1, 20, lb, ub, [], []);
fmin_4 = @(f, X)fmin_ps(f, 2, 10, lb, ub);
fmin = fmin_3;

% Kernels
a = sym('a', [1, 2], 'real');
b = sym('b', [1, 2], 'real');
k_1 = (1 + (a - b) * (a - b)') .^ (-0.5);
k_2 = (1 + log(1 + (a - b) * (a - b)')) .^ (-0.5);
k_3 = symdef_k2c(1, -0.5);
k = k_2;

% Set renderer
fdraw_1 = [];
fdraw_2 = @(f, xNew)fdraw_2d(f, xNew, lb, ub, [nStep, nStep]);
fdraw = fdraw_1;

% (1) Two-component Gaussian mixture
if testCase == 1
    [Mu, C, w] = gmparam_2c();
    fp = @(X)fp_gaussmix(X, Mu, C, w);
    fu = @(X)fscr_gaussmix(X, Mu, C, w);
    tic;
    [X, s, nEval] = smplr(2, fu, k, fmin, nIter, fdraw);
    toc;
end

% (2) Random Gaussian mixture
if testCase == 2
    rng(8);
    [Mu, C, w] = gmparam_rnd();
    rng('shuffle');
    fp = @(X)fp_gaussmix(X, Mu, C, w);
    fu = @(X)fscr_gaussmix(X, Mu, C, w);
    tic;
    [X, s, nEval] = smplr(2, fu, k, fmin, nIter, fdraw);
    toc;
end

% Plot the contours and point set
figure();
t1 = linspace(lb(1), ub(1), nStep)';
t2 = linspace(lb(2), ub(2), nStep)';
T = [repelem(t1, nStep), repmat(t2, nStep, 1)];
p = fp(T);
Z = reshape(p, nStep, nStep);
contour(t1, t2, Z, 'levelstep', 0.001, 'linewidth', 1);
xlabel('x_1');
ylabel('x_2');
title('Target density');
hold on;
plot(X(:, 1), X(:, 2), '+r', 'linewidth', 1);
