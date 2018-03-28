function XCell = cumset(X)
% XCell = cumset(X) converts a point-sequence to cumulative point-sets.
%
% Input:
% X     - nObs-by-nDim matrix of a point-sequence.
%
% Output:
% XCell - nObs-by-1 cell-array of cumulative point-sets.
%
% Date: February 26, 2018

    n = size(X, 1);
    XCell = cell(n, 1);
    for i = 1:n
        XCell{i} = X(1:i, :);
    end
end
