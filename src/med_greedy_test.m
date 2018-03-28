%%
% File: med_greedy_test.m
% Purpose:
% Runs the greedy MED algorithm on various target distributions.
% Date: January 25, 2018
%%

% Test case
testCase = 2;

% Algorithm configurations
smplr = @med_greedy;
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

% (1) Normal
if testCase == 1
    fp = @(X)mvnpdf(X, [0, 0], eye(2));
    tic;
    [X, e, nEval] = smplr(2, fp, fmin, 8, nIter);
    toc;
end

% (2) Random Gaussian mixture
if testCase == 2
    rng(8);
    [Mu, C, w] = gmparam_rnd();
    rng('shuffle');
    fp = @(X)fp_gaussmix(X, Mu, C, w);
    tic;
    [X, e, nEval] = smplr(2, fp, fmin, 8, nIter);
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
