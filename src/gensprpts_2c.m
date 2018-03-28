%%
% File: gensprpts_2c.m
% Purpose:
% Generates "super" point plots for various discrete approximation algorithms.
% The target is a two-component Gaussian mixture distribution.
% Date: February 29, 2018
%%

% Target density and score functions
[Mu, C, w] = gmparam_2c();
fp = @(X)fp_gaussmix(X, Mu, C, w);
fu = @(X)fscr_gaussmix(X, Mu, C, w);

% Symbolic variables
a = sym('a', [1, 2], 'real');
b = sym('b', [1, 2], 'real');

% Optimisers
lb = [-5, -5];
ub = [5, 5];
mu0 = [0, 0];
Sigma0 = diag([25, 25]);
nStep = 100;
fmin_1 = @(f, X)fmin_adares(f, X, 3, mu0, Sigma0, 1, 20, lb, ub, [], []);
fmin_2 = @(f, X)fmin_adamc(f, X, 20, mu0, Sigma0, 1, 20, lb, ub, [], []);
fmin_3 = @(f, X)fmin_gsrtn(f, X, lb, ub, [nStep, nStep]);
fmin_4 = @(f, X)fmin_gs2d(f, lb, ub, [nStep, nStep]);
fmin_5 = @(f, X)fmin_ps(f, 2, 10, lb, ub);
fmin = fmin_2;

% Extensible point sets
nPart_1 = 8000;
nPart_2 = 500;
nPart_3 = 500;
nPart_4 = 500;

% (1) Monte Carlo
X = gaussmix_rnd(Mu, C, w, nPart_1);
nEval = ones(nPart_1, 1);
X_1 = cumset(X);
nEval_1 = log(cumsum(nEval));

% (2) Stein Greedy
k = (4 + log(1 + (a - b) * (a - b)')) .^ (-1);
[X, ~, nEval] = stein_greedy(2, fu, k, fmin, nPart_2, []);
X_2 = cumset(X);
nEval_2 = log(cumsum(nEval));

% (3) Stein Herding
k = (0.5 + (a - b) * (a - b)') .^ (-0.7);
[X, ~, nEval] = stein_fw(2, fu, k, fmin, nPart_3, []);
X_3 = cumset(X);
nEval_3 = log(cumsum(nEval));

% (4) MED
[X, ~, nEval] = med_greedy(2, fp, fmin, 4, nPart_4);
X_4 = cumset(X);
nEval_4 = log(cumsum(nEval));

% (5) Stein Co-Descent
nPart_5 = 100;
nIter_5 = 6 .* nPart_5;
k = (2 + log(1 + (a - b) * (a - b)')) .^ (-1);
X0 = bx([0, 0], 10, nPart_5);
[X, nEval] = stein_codescent(fu, k, fmin, X0, nIter_5, []);
X_5 = num2cell(X, [1, 2]);
X_5 = X_5(:);
nEval_5 = log(cumsum(nEval));

% (6) SVGD
nPart_6 = 100;
nIter_6 = 200;
fstp = @(i, Phi)fstp_adagrad(i, Phi, 0.1, 0.9);
k = (0.5 + (a - b) * (a - b)') .^ (-0.9);
X0 = bx([0, 0], 10, nPart_6);
[X, nEval] = mysvgd(fu, k, X0, fstp, nIter_6);
X_6 = num2cell(X, [1, 2]);
X_6 = X_6(:);
nEval_6 = log(cumsum(nEval));

% Prepare for contour plots
t1 = linspace(lb(1), ub(1), nStep)';
t2 = linspace(lb(2), ub(2), nStep)';
T = [repelem(t1, nStep), repmat(t2, nStep, 1)];
p = fp(T);
Z = reshape(p, nStep, nStep);

% Generate plots
figure();
axis([5.8, 10.2, -0.2, 7.2]);
set(gca, 'fontsize', 9);
xlabel('log n_{eval}', 'fontsize', 11);
yticks(0.5:1.2:6.5);
yticklabels({ ...
    'SVGD', ...
    'Stein Co-Des.', ...
    'MED', ...
    'Stein Herd.', ...
    'Stein Greedy', ...
    'Monte Carlo'})
ytickangle(90);
main = gca;
lev = 6:-1.2:0;
alg = {'1', '2', '3', '4', '5', '6'};
ref = [6, 7.5, 9];
for i = 1:numel(alg)
    for j = 1:numel(ref)
        XCell = eval(['X_', alg{i}]);
        nEval = eval(['nEval_', alg{i}]);
        [~, m] = min(abs(nEval - ref(j)));
        axes('position', getpos(main, nEval(m), lev(i), 1, 1));
        contour(t1, t2, Z, 'levelstep', 0.005, 'linewidth', 0.5);
        X = XCell{m};
        hold on;
        plot(X(:, 1), X(:, 2), '.r', 'markersize', 4);
        axis([lb(1), ub(1), lb(2), ub(2)]);
        set(gca, 'xtick', []);
        set(gca, 'ytick', []);
    end
end

% Window setting
set(gcf, 'renderer', 'painters');
set(gcf, 'units', 'centimeters');
set(gcf, 'position', [0.5, 1.5, 10, 17.5]);

% Print setting
set(gcf, 'paperunits', 'centimeters');
set(gcf, 'paperpositionmode', 'manual');
set(gcf, 'paperposition', [0, 0, 10, 17.5]);
set(gcf, 'papertype', '<custom>');
set(gcf, 'papersize', [10, 17.5]);

% Print to PDF
name = sprintf('gensprpts_2c');
print(name, '-dpdf');
