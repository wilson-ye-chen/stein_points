function [xMin, dMin, fMin, nEval] = fmin_adamc( ...
    f, X, nMc, mu0, Sigma0, sigsq, delay, lb, ub, A, b)
% [xMin, dMin, fMin, nEval] = fmin_adamc(f, X, nMc, mu0, Sigma0, sigsq, ...
% delay, lb, ub, A, b) minimises the objective function using an adaptive
% Monte Carlo optimisation approach where the proposal distributions is
% constructed adaptively using a mixture of Gaussians based on the current
% points.
%
% Date: October 3, 2017

    % Adapt proposal based on X
    [nObs, nDim] = size(X);
    if nObs <= delay
        frnd = @()mvnrnd(mu0, Sigma0, 1);
    else
        Sigma = sigsq .* eye(nDim);
        frnd = @()mvnrnd(X(randi(nObs), :), Sigma, 1);
    end

    % Generate feasible draws
    XMc = genfeasible(frnd, nMc, nDim, lb, ub, A, b);

    % Evaluate vectorised objective
    [fX, D] = f(XMc);
    [fMin, iMin] = min(fX);
    xMin = XMc(iMin, :);
    dMin = D(iMin, :);
    nEval = nMc;
end

function X = genfeasible(frnd, n, nDim, lb, ub, A, b)
    X = zeros(n, nDim);
    for i = 1:n
        t = frnd();
        while ~isfeasible(t, lb, ub, A, b)
            t = frnd();
        end
        X(i, :) = t;
    end
end

function tf = isfeasible(x, lb, ub, A, b)
    if ~isempty(lb) && any(x < lb)
        tf = false;
        return
    end
    if ~isempty(ub) && any(x > ub)
        tf = false;
        return
    end
    if ~isempty(A) && any(A * x > b)
        tf = false;
        return
    end
    tf = true;
end
