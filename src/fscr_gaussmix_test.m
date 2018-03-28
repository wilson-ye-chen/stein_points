%%
% File: fscr_gaussmix_test.m
%
% Purpose:
% Generates a point set X that approximates a randomly constructed Gaussian
% mixture distribution in 2 dimensions.
%
% Date: January 2, 2018
%%

% Construct a random Gaussian mixture.
rng(7);
[Mu, S, w] = gmparam_rnd();
rng('shuffle');
fp = @(X)fp_gaussmix(X, Mu, S, w);

% Gaussian mixture score function
fu = @(X)fscr_gaussmix(X, Mu, S, w);

% Number of points
nPt = 200;

% Symbolic variables
a = sym('a', [1, 2], 'real');
b = sym('b', [1, 2], 'real');

% IMQ kernel
k = (1 + (a - b) * (a - b)') .^ (-0.5);

% Generate point-sequence using AdaMc-IMQ
mu0 = [0, 0];
Sigma0 = diag([25, 25]);
lb = [-10, -10];
ub = [10, 10];
nStep = 100;
% fmin = @(f, X)fmin_gs2d(f, lb, ub, [nStep, nStep]);
fmin = @(f, X)fmin_adares(f, X, 3, mu0, Sigma0, 1, 20, lb, ub, [], []);
% fmin = @(f, X)fmin_adamc(f, X, 20, mu0, Sigma0, 1, 20, lb, ub, [], []);
fprintf('Stein-IMQ-%d...\n', nPt);
tic;
[X, s, nEval] = stein_greedy(2, fu, k, fmin, nPt, []);
toc;

% Get associated scores
D = fu(X);

% Compute numerical scores for debugging
DNum = numdiff(2, X, @(T)log(fp(T)), 1e-6);
figure();
plot(D, 'x');
hold on;
plot(DNum, 'o');

% Plot the target (properly normalised in this case).
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

% Superimpose the particular point set and associated scores
hold on;
quiver(X(:, 1), X(:, 2), D(:, 1), D(:, 2), 'o');
