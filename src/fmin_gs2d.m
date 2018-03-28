function [xMin, dMin, fMin, nEval] = fmin_gs2d(f, lb, ub, n)
% [xMin, dMin, fMin, nEval] = fmin_gs2d(f, lb, ub, n) minimises the bivariate
% function f using a grid-search.
%
% Input:
% f     - function handle to the objective. This function must return
%         the score matrix of the target density as its second output.
% lb    - 2-vector of lower bounds of the search grid.
% ub    - 2-vector of upper bounds of the search grid.
% n     - 2-vector of grid resolutions.
%
% Output:
% xMin  - 1-by-2 vector that minimises f.
% dMin  - 1-by-2 score vecotr at xMin.
% fMin  - objective value at xMin.
% nEval - number of function evaluations.
%
% Date: September 19, 2017

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
