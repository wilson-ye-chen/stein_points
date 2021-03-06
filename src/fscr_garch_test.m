%%
% File: fscr_garch_test.m
%
% Purpose:
% Consider a particular point set X, where each row of X represents one
% set of parameter values for a GARCH model. We are interested in the
% posterior over these parameters. In particular we want to approximate
% the posterior with a point set generated by AdaMC.
%
% Date: October 28, 2017
%%

% Load S&P 500 return data
load('data_spx.mat');

% GARCH score function
h1 = var(r);
fu = @(X)fscr_garch(X, r, h1);

% Number of points
nPt = 100;

% Symbolic variables
a = sym('a', [1, 3], 'real');
b = sym('b', [1, 3], 'real');

% IMQ kernel
k = (1 + (a - b) * (a - b)') .^ (-0.5);

% Generate point-sequence using AdaMc-IMQ
mu0 = [-4, -2, -0.1];
Sigma0 = diag([0.5, 0.5, 0.5]);
lb = [-6.5, -3.5, -0.5];
ub = [-2, -0.5, -0.01];
fmin = @(f, X)fmin_adares(f, X, 3, mu0, Sigma0, 0.2, 20, lb, ub, [], []);
% fmin = @(f, X)fmin_adamc(f, X, 100, mu0, Sigma0, 0.2, 20, lb, ub, [], []);
fprintf('AdaMC-IMQ-%d...\n', nPt);
tic;
[X, s, nEval] = stein_greedy(3, fu, k, fmin, nPt, []);
toc;

% Get associated scores for x2 and x3 while fixing x1
x1 = mean(X(:, 1));
X23 = X;
X23(:, 1) = x1;
D = fscr_garch(X23, r, h1);

% Plot the (un-normalised) log-target
figure()
m = 100;
t1 = linspace(lb(2), ub(2), m)';
t2 = linspace(lb(3), ub(3), m)';
T = [x1 .* ones(m .^ 2, 1), repelem(t1, m), repmat(t2, m, 1)];
p = fp_garch(T, r, h1);
Z = reshape(p, m, m);
contour(t1, t2, Z, 'levelstep', 10);
xlabel('x_2');
ylabel('x_3');
title('Log-target + constant');

% Superimpose the particular point set and associated scores
hold on;
quiver(X(:, 2), X(:, 3), D(:, 2), D(:, 3), 'o');
