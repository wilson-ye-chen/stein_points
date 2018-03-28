%%
% File: fscr_igarch_test.m
%
% Purpose:
% Consider a particular point set X, where each row of X represents one
% set of parameter values for a IGARCH model. We are interested in the
% posterior over these parameters. In particular we want to approximate
% the posterior with a point set generated by AdaMC.
%
% Date: November 6, 2017
%%

% Load S&P 500 return data
load('data_spx.mat');
str = 2501;
wdt = 2000;
r = r(str:(str + wdt - 1));

% GARCH score function
h1 = var(r);
fu = @(X)fscr_igarch(X, r, h1);

% Number of points
nPt = 200;

% Symbolic variables
a = sym('a', [1, 2], 'real');
b = sym('b', [1, 2], 'real');

% IMQ kernel
k = (1e-5 + (a - b) * (a - b)') .^ (-0.5);

% Generate point-sequence using AdaMc-IMQ
mu0 = [0.02, 0.1];
Sigma0 = diag([1e-4, 1e-3]);
lb = [1e-6, 1e-6];
ub = [0.05, 0.25];
nStep = 500;
% fmin = @(f, X)fmin_gs2d(f, lb, ub, [nStep, nStep]);
% fmin = @(f, X)fmin_adares(f, X, 3, mu0, Sigma0, 1e-5, 5, lb, ub, [], []);
fmin = @(f, X)fmin_adamc(f, X, 10, mu0, Sigma0, 1e-5, 5, lb, ub, [], []);
fprintf('AdaMC-IMQ-%d...\n', nPt);
tic;
[X, s, nEval] = stein_greedy(2, fu, k, fmin, nPt, []);
toc;

% Get associated analytical and numerical scores for x1 and x2
D = fscr_igarch(X, r, h1);
DNum = numdiff(2, X, @(T)fp_igarch(T, r, h1), 1e-6);
figure();
plot(D, 'x');
hold on;
plot(DNum, 'o');

% Plot the (un-normalised) log-target
figure();
m = 100;
t1 = linspace(lb(1), ub(1), m)';
t2 = linspace(lb(2), ub(2), m)';
T = [repelem(t1, m), repmat(t2, m, 1)];
p = fp_igarch(T, r, h1);
Z = reshape(p, m, m);
contour(t1, t2, Z, 'levelstep', 5);
xlabel('x_1');
ylabel('x_2');
title('Log-target + constant');

% Superimpose the particular point set and associated scores
hold on;
quiver(X(:, 1), X(:, 2), D(:, 1), D(:, 2), 'o');