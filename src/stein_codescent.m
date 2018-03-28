function [X, nEval] = stein_codescent(fscr, k, fmin, X0, nIter, fdraw)
% [X, nEval] = stein_codescent(fscr, k, fmin, X0, nIter, fdraw) generates
% a point-set that approximates the density whose score is given by fscr.
%
% Input:
% fscr  - handle to the score function of the target density. The score
%         function must accept either an 1-by-nDim row vector or a n-by
%         -nDim matrix. It returns either an 1-by-nDim row vector or a
%         n-by-nDim matrix.
% k     - symbolic expression of the kernel k(a,b), where a and b are 1
%         -by-nDim row vectors. It is important that the argument names
%         are literally "a" and "b".
% fmin  - function handle to a nDim-dimensional minimiser.
% X0    - nPar-by-nDim matrix of the initial point-set.
% nIter - number of iterations.
% fdraw - function handle for plotting the objective at each iteration.
%         Plotting is disabled if fdraw is set to [].
%
% Output:
% X     - nPar-by-nDim-by-nIter array of generated points.
% nEval - number of score function evaluations at each iteration.
%
% Date: January 22, 2018

    % Symbolic computations
    [nPar, nDim] = size(X0);
    a = sym('a', [1, nDim], 'real');
    b = sym('b', [1, nDim], 'real');
    dka = sym(zeros(1, nDim));
    dkb = sym(zeros(1, nDim));
    d2k = sym(zeros(1, nDim));
    for i = 1:nDim
        dka(i) = gradient(k, a(i));
        dkb(i) = gradient(k, b(i));
        d2k(i) = gradient(dka(i), b(i));
    end

    % Generate MATLAB code
    matlabFunction(k, 'vars', {a, b}, 'file', 'fk.m');
    matlabFunction(dka, 'vars', {a, b}, 'file', 'fdka.m');
    matlabFunction(dkb, 'vars', {a, b}, 'file', 'fdkb.m');
    matlabFunction(d2k, 'vars', {a, b}, 'file', 'fd2k.m');

    % Outputs
    X = zeros(nPar, nDim, nIter);
    nEval = zeros(nIter, 1);

    % Initial points
    X(:, :, 1) = X0;
    D = fscr(X0);
    nEval(1) = nPar;

    % Special case: only one initial point is given
    if nPar == 1
        f = @(XNew)fk0aa(XNew, fscr);
        emp = double.empty(0, nDim);
        for i = 2:nIter
            [X(1, :, i), ~, ~, nEval(i)] = fmin(f, emp);
            fprintf('i = %d\n', i);
            if ~isempty(fdraw)
                fdraw(f, X(1, :, i));
            end
        end
        return
    end

    % Move one point at a time
    M = logical(eye(nPar));
    for i = 2:nIter
        j = M(mod(i, nPar) + 1, :);
        f = @(XNew)fps(XNew, fscr, X(~j, :, i - 1), D(~j, :), nPar);
        X(:, :, i) = X(:, :, i - 1);
        [X(j, :, i), D(j, :), ~, nEval(i)] = fmin(f, X(~j, :, i - 1));
        fprintf('i = %d\n', i);
        if ~isempty(fdraw)
            fdraw(f, X(j, :, i));
        end
    end
end

function [ps, DNew] = fps(XNew, fscr, X, D, n)
% Input:
% XNew - nNew-by-nDim matrix of new points.
% fscr - handle to the score function of the target density.
% X    - nObs-by-nDim matrix of current sample.
% D    - nObs-by-nDim matrix of scores at X.
% n    - observation index of XNew.
%
% Output:
% ps   - nNew-by-1 column vector of partial sums for XNew.
% DNew - nNew-by-nDim matrix of scores at XNew.

    nNew = size(XNew, 1);
    DNew = fscr(XNew);
    A = repmat(XNew, n - 1, 1);
    B = repelem(X(1:(n - 1), :), nNew, 1);
    Da = repmat(DNew, n - 1, 1);
    Db = repelem(D(1:(n - 1), :), nNew, 1);
    K0ab = reshape(fk0(A, B, Da, Db), nNew, []);
    k0aa = fk0(XNew, XNew, DNew, DNew);
    ps = sum(K0ab, 2) .* 2 + k0aa;
end

function [k0, DNew] = fk0aa(XNew, fscr)
% Input:
% XNew - nNew-by-nDim matrix of new points.
% fscr - handle to the score function of the target density.
%
% Output:
% k0   - nNew-by-1 column vector of Stein kernel values.
% DNew - nNew-by-nDim matrix of scores at XNew.

    DNew = fscr(XNew);
    k0 = fk0(XNew, XNew, DNew, DNew);
end

function k0 = fk0(A, B, Da, Db)
% Input:
% A  - m-by-nDim matrix of the first arguments.
% B  - m-by-nDim matrix of the second arguments.
% Da - m-by-nDim matrix of scores at A.
% Db - m-by-nDim matrix of scores at B.
%
% Output:
% k0 - m-by-1 column vector of Stein kernel values.

    nDim = size(A, 2);
    K0i = ...
        fd2k(A, B) + ...
        Da .* fdkb(A, B) + ...
        Db .* fdka(A, B) + ...
        Da .* Db .* repmat(fk(A, B), 1, nDim);
    k0 = sum(K0i, 2);
end
