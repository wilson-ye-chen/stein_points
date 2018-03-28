function [X, e, nEval] = med_greedy(nDim, fpdf, fmin, k, nIter)
% [X, e, nEval] = med_greedy(nDim, fpdf, fmin, k, nIter) generates a point-
% set called minimum energy design (MED) using a one-point-at-a-time greedy
% algorithm described by Joseph et al (2017).
%
% Input:
% nDim  - number of dimensions of the target density.
% fpdf  - handle to the target density function.
% fmin  - handle to a nDim-dimensional minimiser.
% k     - power parameter of the generalised energy criterion.
% nIter - length of the generated sequence of points.
%
% Output:
% X     - nIter-by-nDim matrix of generated points.
% e     - minimised energy at each iteration.
% nEval - number of density evaluations at each iteration.
%
% Date: January 25, 2018

    X = zeros(nIter, nDim);
    y = zeros(nIter, 1);
    e = zeros(nIter, 1);
    nEval = zeros(nIter, 1);

    % Generate x_1
    f = @(XNew)fq(XNew, fpdf);
    [X(1, :), y(1), e(1), nEval(1)] = fmin(f, double.empty(0, nDim));
    fprintf('n = 1\n');

    % Generate the rest
    for n = 2:nIter
        f = @(XNew)fe(XNew, fpdf, k, X, y, n);
        [X(n, :), y(n), e(n), nEval(n)] = fmin(f, X(1:(n - 1), :));
        fprintf('n = %d\n', n);
    end
end

function [e, yNew] = fe(XNew, fpdf, k, X, y, n)
    [nNew, nDim] = size(XNew);
    yNew = fpdf(XNew) .^ (k ./ (2 .* nDim));
    A = repmat(XNew, n - 1, 1);
    B = repelem(X(1:(n - 1), :), nNew, 1);
    yb = repelem(y(1:(n - 1)), nNew, 1);
    t = 1 ./ (yb .* fd(A, B) .^ k);
    e = 1 ./ yNew .* sum(reshape(t, nNew, []), 2);
end

function [q, yNew] = fq(XNew, fpdf)
    nDim = size(XNew, 2);
    yNew = fpdf(XNew) .^ (1 ./ (2 .* nDim));
    q = 1 ./ yNew;
end

function d = fd(A, B)
    d = sqrt(sum((A - B) .^ 2, 2));
end
