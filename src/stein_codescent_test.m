%%
% File: stein_codescent_test.m
% Purpose:
% Runs the coordinate-descent Stein sampling algorithm on various target
% distributions.
% Date: January 3, 2018
%%

% Test case
testCase = 2;

% Algorithm configurations
smplr = @stein_codescent;
nPart = 50;
nIter = 6 .* nPart;

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
fmin = fmin_1;

% Kernels
a = sym('a', [1, 2], 'real');
b = sym('b', [1, 2], 'real');
k_1 = (1 + (a - b) * (a - b)') .^ (-0.5);
k_2 = (4 + log(1 + (a - b) * (a - b)')) .^ (-1);
k = k_1;

% Set renderer
fdraw_1 = [];
fdraw_2 = @(f, xNew)fdraw_2d(f, xNew, lb, ub, [nStep, nStep]);
fdraw = fdraw_1;

% Initial particles
X0_1 = unifrnd(-10, 10, nPart, 2);
X0_2 = mvnrnd([7, 0], eye(2), nPart);
X0_3 = circ([0, 0], 0.1, nPart);
X0_4 = bx([0, 0], 20, nPart);
X0 = X0_4;

% (1) Normal
if testCase == 1
    y = sym('y', [1, 2], 'real');
    p = -0.5 * y * y';
    u = gradient(p, y);
    fp = @(X)mvnpdf(X, [0, 0], eye(2));
    fu = matlabFunction(u', 'vars', {y});
    tic;
    [X, nEval] = smplr(fu, k, fmin, X0, nIter, fdraw);
    toc;
end

% (2) Random Gaussian mixture
if testCase == 2
    rng(7);
    [Mu, C, w] = gmparam_rnd();
    rng('shuffle');
    fp = @(X)fp_gaussmix(X, Mu, C, w);
    fu = @(X)fscr_gaussmix(X, Mu, C, w);
    tic;
    [X, nEval] = smplr(fu, k, fmin, X0, nIter, fdraw);
    toc;
end

% Evaluate log-density of the target over a grid
t1 = linspace(lb(1), ub(1), nStep)';
t2 = linspace(lb(2), ub(2), nStep)';
T = [repelem(t1, nStep), repmat(t2, nStep, 1)];
p = fp(T);
Z = reshape(p, nStep, nStep);

% Write to a video file
fprintf('Writing to video file...\n');
v = VideoWriter('stein_codescent_out.avi');
v.FrameRate = 10;
v.Quality = 50;
open(v);
for i = 1:nIter
    contour(t1, t2, Z, 'levelstep', 0.001, 'linewidth', 1);
    xlabel('x_1');
    ylabel('x_2');
    title('Target density');
    hold on;
    plot(X(:, 1, i), X(:, 2, i), '+r', 'linewidth', 1);
    hold off;
    frame = getframe(gcf);
    writeVideo(v, frame);
end
close(v);
