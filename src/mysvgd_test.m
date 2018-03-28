%%
% File: testmysvgd.m
% Purpose:
% Runs the SVGD algorithm with custom kernels to approximate various target
% distributions.
% Date: January 10, 2018
%%

% Test case
testCase = 2;

% Algorithm configurations
nPart = 100;
nIter = 200;

% Kernels
a = sym('a', [1, 2], 'real');
b = sym('b', [1, 2], 'real');
k_1 = exp(-((a - b) * (a - b)') ./ 1);
k_2 = (2 + (a - b) * (a - b)') .^ (-0.5);
k_3 = (4 + log(1 + (a - b) * (a - b)')) .^ (-1);
k = k_2;

% Step-size function
fstp = @(i, Phi)fstp_adagrad(i, Phi, 0.1, 0.9);

% Initial particles
X0_1 = unifrnd(-10, 10, nPart, 2);
X0_2 = mvnrnd([7, 0], eye(2), nPart);
X0_3 = circ([0, 0], 0.1, nPart);
X0_4 = bx([0, 0], 0.1, nPart);
X0 = X0_1;

% (1) Normal
if testCase == 1
    y = sym('y', [1, 2], 'real');
    p = -0.5 * y * y';
    u = gradient(p, y);
    fp = @(X)mvnpdf(X, [0, 0], eye(2));
    fu = matlabFunction(u', 'vars', {y});
    tic;
    [X, nEval] = mysvgd(fu, k, X0, fstp, nIter);
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
    [X, nEval] = mysvgd(fu, k, X0, fstp, nIter);
    toc;
end

% Plot the contours and point set
lb = [-10, -10];
ub = [10, 10];
nStep = 100;
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
plot(X(:, 1, end), X(:, 2, end), '+r', 'linewidth', 1);
