function D = numdiff(nDim, X, f, dlt)
% D = numdiff(nDim, X, f, dlt) computes the numerical gradients of an input
% function over a point-set. Central differencing is used.
%
% Input:
% nDim - number of dimensions.
% X    - n-by-nDim matrix of points.
% f    - handle to a vectorised input function. This function should accept
%        a n-by-nDim matrix as its argument, and returns a n-by-1 vector.
% dlt  - differencing step.
%
% Output:
% D    - n-by-nDim matrix of gradients.
%
% Date: November 6, 2917

    for i = 1:nDim
        T = repmat(X, 2, 1);
        T(:, i) = [X(:, i) - dlt; X(:, i) + dlt];
        y = f(T);
        D(:, i) = diff(reshape(y, [], 2), 1, 2) ./ (2 .* dlt);
    end
end
