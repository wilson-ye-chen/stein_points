function [xMin, dMin, fMin, nEval] = fmin_gsrtn(f, X, lb, ub, n0)
% [xMin, dMin, fMin, nEval] = fmin_gsrtn(f, X, lb, ub, n0) minimises the
% bivariate function f using a grid-search, where the grid size grows with
% the size of X.
%
% Input:
% f     - function handle to the objective. This function must return
%         the score matrix of the target density as its second output.
% X     - nObs-by-2 matrix of points.
% lb    - 2-vector of lower bounds of the search grid.
% ub    - 2-vector of upper bounds of the search grid.
% n0    - 2-vector of initial grid resolutions.
%
% Output:
% xMin  - 1-by-2 vector that minimises f.
% dMin  - 1-by-2 score vecotr at xMin.
% fMin  - objective value at xMin.
% nEval - number of function evaluations.
%
% Date: January 24, 2018

    n = n0 + round(sqrt(size(X, 1) + 1));
    sx1 = linspace(lb(1), ub(1), n(1));
    sx2 = linspace(lb(2), ub(2), n(2));
    [X1, X2] = meshgrid(sx1, sx2);
    X = [X1(:), X2(:)];
    [fX, D] = f(X);
    [fMin, iMin] = min(fX);
    xMin = X(iMin, :);
    dMin = D(iMin, :);
    nEval = n(1) .* n(2);
end
