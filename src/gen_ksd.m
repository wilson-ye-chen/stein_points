%%
% File: gen_ksd.m
% Purpose:
% Generates KSD plots for simulation studies.
% Date: January 5, 2018
%%

% Number of particles
nPart = 100;

% Target density and score functions
rng(7);
[Mu, C, w] = gmparam_rnd();
rng('shuffle');
fp = @(X)fp_gaussmix(X, Mu, C, w);
fu = @(X)fscr_gaussmix(X, Mu, C, w);

% Symbolic variables
a = sym('a', [1, 2], 'real');
b = sym('b', [1, 2], 'real');

% IMQ kernel
k = (1 + (a - b) * (a - b)') .^ (-0.5);

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

% IID draws from target
if exist('X_0') ~= 1
    X_0 = gaussmix_rnd(Mu, C, w, nPart);
    s_0 = ksd(X_0, fu, k);
end

% Run AdaRes
if exist('X_1') ~= 1
    fprintf('AdaRes-Stein-%d...\n', nPart);
    tic;
    [X_1, s_1, nEval_1] = stein_greedy(2, fu, k, fmin_3, nPart, []);
    toc;
end

% Run AdaMC
if exist('X_2') ~= 1
    fprintf('AdaMC-Stein-%d...\n', nPart);
    tic;
    [X_2, s_2, nEval_2] = stein_greedy(2, fu, k, fmin_2, nPart, []);
    toc;
end

% Run SVGD
if exist('X_3') ~= 1
    nIter = 200;
    X0 = unifrnd(-10, 10, nPart, 2);
    fstp = @(i, Phi)fstp_adagrad(i, Phi, 0.1, 0.9);
    fprintf('SVGD-IMQ-%d...\n', nPart);
    tic;
    [X_3, nEval_3] = mysvgd(fu, k, X0, fstp, nIter);
    toc;

    % Compute KSD-IMQ
    fprintf('KSD-IMQ for SVGD-IMQ-%d...\n', nPart);
    tic;
    s_3 = zeros(nIter, 1);
    for i = 1:nIter
        s = ksd(X_3(:, :, i), fu, k);
        s_3(i) = s(end);
    end
    toc;
end

% Run AdaRes-MED
if exist('X_4') ~= 1
    fprintf('AdaRes-MED-%d...\n');
    tic;
    [X_4, ~, nEval_4] = med_greedy(2, fp, fmin_3, 1, nPart);
    toc;
    s_4 = ksd(X_4, fu, k);
end

% Run AdaMC-MED
if exist('X_5') ~= 1
    fprintf('AdaMC-MED-%d...\n');
    tic;
    [X_5, ~, nEval_5] = med_greedy(2, fp, fmin_2, 1, nPart);
    toc;
    s_5 = ksd(X_5, fu, k);
end

% Generate plots
alg = {'1', '2', '3', '4', '5'};
mkr = {'x', 'd', 'v', 'o', 's'};
for i = 1:5
    xx = log(cumsum(eval(['nEval_', alg{i}])));
    yy = log(eval(['s_', alg{i}]));
    plot(xx, yy, mkr{i}, 'MarkerSize', 5);
    hold on;
end
plot(xlim, log([s_0(end), s_0(end)]), 'k--');
set(gca, 'ticklabelinterpreter', 'latex', 'fontsize', 11);
xlabel('Log of the number of score evaluations', ...
    'interpreter', 'latex', ...
    'fontsize', 11);
ylabel('Log of the KSD-IMQ', ...
    'interpreter', 'latex', ...
    'fontsize', 11);
h = legend( ...
    'AdaRes-Stein', ...
    'AdaMC-Stein', ...
    'SVGD', ...
    'AdaRes-MED', ...
    'AdaMC-MED', ...
    'IID Sample', ...
    'location', 'northeast');
set(h, 'interpreter', 'latex', 'fontsize', 7);

% Window setting
set(gcf, 'renderer', 'opengl');
set(gcf, 'units', 'centimeters');
set(gcf, 'position', [0.5, 1.5, 18, 9]);

% Print setting
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'paperpositionmode', 'manual');
set(gcf, 'paperposition', [0, 0, 18, 9]);
set(gcf, 'papertype', '<custom>');
set(gcf, 'papersize', [18, 9]);

% Print to PNG
name = sprintf('ksd_vs_eval_%d', nPart);
print(name, '-r300', '-dpng');
