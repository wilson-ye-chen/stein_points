%%
% File: svgd_test.m
% Purpose:
% Runs the SVGD algorithm on various target distributions.
% Date: October 29, 2017
%%

% Case flags
case_1 = false;
case_2 = true;
case_3 = false;
case_4 = false;

% Algorithm configurations
nPart = 100;
nIter = 500;

% (1) Normal
if case_1
    y = sym('y', [2, 1], 'real');
    p = -0.5 * y' * y;
    u = gradient(p, y);
    fu = matlabFunction(u);
    X0 = mvnrnd([-10, 0], eye(2), nPart);
    tic;
    X_1_svgd = svgd(X0, @(X)reshape(fu(X(:, 1), X(:, 2)), [], 2), nIter);
    toc;
end

% (2) Correlated mixture
if case_2
    y = sym('y', [2, 1], 'real');
    m1 = sym([-1.5; 0]);
    m2 = sym([1.5; 0]);
    C1 = sym([1, 0.8; 0.8, 1]);
    C2 = sym([1, -0.8; -0.8, 1]);
    H1 = inv(C1);
    H2 = inv(C2);
    p = log( ...
        exp(-0.5 * (y - m1)' * H1 * (y - m1)) + ...
        exp(-0.5 * (y - m2)' * H2 * (y - m2)));
    u = gradient(p, y);
    fu = matlabFunction(u);
    X0 = mvnrnd([-10, 0], eye(2), nPart);
    tic;
    X_2_svgd = svgd(X0, @(X)reshape(fu(X(:, 1), X(:, 2)), [], 2), nIter);
    toc;
end

% (3) Far-modal
if case_3
    y = sym('y', [2, 1], 'real');
    m1 = sym([-4; 0]);
    m2 = sym([4; 0]);
    p = log( ...
        exp(-0.5 * (y - m1)' * (y - m1)) + ...
        exp(-0.5 * (y - m2)' * (y - m2)));
    u = gradient(p, y);
    fu = matlabFunction(u);
    X0 = mvnrnd([-10, 0], eye(2), nPart);
    tic;
    X_3_svgd = svgd(X0, @(X)reshape(fu(X(:, 1), X(:, 2)), [], 2), nIter);
    toc;
end

% (4) Far-modal, E(X0)=(0,0)
if case_4
    y = sym('y', [2, 1], 'real');
    m1 = sym([-4; 0]);
    m2 = sym([4; 0]);
    p = log( ...
        exp(-0.5 * (y - m1)' * (y - m1)) + ...
        exp(-0.5 * (y - m2)' * (y - m2)));
    u = gradient(p, y);
    fu = matlabFunction(u);
    X0 = mvnrnd([0, 0], eye(2), nPart);
    tic;
    X_4_svgd = svgd(X0, @(X)reshape(fu(X(:, 1), X(:, 2)), [], 2), nIter);
    toc;
end
