function [xMin, dMin, fMin, nEval, fLoc] = fmin_adares( ...
    f, X, nRes, mu0, Sigma0, sigsq, delay, lb, ub, A, b)
% [xMin, dMin, fMin, nEval, fLoc] = fmin_adares(f, X, nRes, mu0, Sigsq0, ...
% sigsq, delay, lb, ub, A, b) minimises the objective function using a
% globalised Nelder-Mead algorithm. Globalisation is achieved by adaptive
% restarts where a local Nelder-Mead optimiser is restarted multiple times,
% with adaptively chosen initial values based on the current sample.
%
% Date: September 28, 2017

    % Adapt proposal based on X
    [nObs, nDim] = size(X);
    if nObs <= delay
        frnd = @()mvnrnd(mu0, Sigma0, 1);
    else
        Sigma = sigsq .* eye(nDim);
        frnd = @()mvnrnd(X(randi(nObs), :), Sigma, 1);
    end

    % Constrained Nelder–Mead method for multiple local searches
    opt = optimset('tolfun', 1e-3, 'tolx', 1e-3, 'display', 'off');
    XLoc = zeros(nRes, nDim);
    fLoc = zeros(nRes, 1);
    nEval = 0;
    X0 = genfeasible(frnd, nRes, nDim, lb, ub, A, b);
    for i = 1:nRes
        [XLoc(i, :), fLoc(i), ~, out] = fminsearchcon( ...
            f, X0(i, :), lb, ub, A, b, [], opt);
        nEval = nEval + out.funcCount;
    end

    % Obtain global solution
    [fMin, iMin] = min(fLoc);
    xMin = XLoc(iMin, :);

    % One last call to the objective to obtain the score vector
    [~, dMin] = f(xMin);
    nEval = nEval + 1;
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
