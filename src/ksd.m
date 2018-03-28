function s = ksd(X, fscr, k)
% s = ksd(X, fscr, k) generates a cumulative sequence of KSD values.
%
% Input:
% X    - nObs-by-nDim matrix of nDim-dimensional points.
% fscr - handle to the score function of the target density. The score
%        function must accept either an 1-by-nDim row vector or a n-by
%        -nDim matrix. It returns either an 1-by-nDim row vector or a
%        n-by-nDim matrix.
% k    - symbolic expression of the kernel k(a,b), where a and b are 1
%        -by-nDim row vectors. It is important that the argument names
%        are literally "a" and "b".
%
% Output:
% s    - nObs-by-1 vector of cumulative KSD values.
%
% Date: October 12, 2017

    % Symbolic computations
    [nObs, nDim] = size(X);
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

    % Scores of the target
    D = fscr(X);

    % For x_1
    s = zeros(nObs, 1);
    ss = fk0(X(1, :), X(1, :), D(1, :), D(1, :));
    s(1) = sqrt(ss);

    % For the rest
    for n = 2:nObs
        ss = ss + fps(X(n, :), D(n, :), X, D, n);
        s(n) = sqrt(ss) ./ n;
    end
end

function ps = fps(XNew, DNew, X, D, n)
% Input:
% XNew - nNew-by-nDim matrix of new points.
% DNew - nNew-by-nDim matrix of scores at XNew.
% X    - nObs-by-nDim matrix of current sample.
% D    - nObs-by-nDim matrix of scores at X.
% n    - observation index of XNew.
%
% Output:
% ps   - nNew-by-1 vector of partial sums for XNew.

    nNew = size(XNew, 1);
    A = repmat(XNew, n - 1, 1);
    B = repelem(X(1:(n - 1), :), nNew, 1);
    Da = repmat(DNew, n - 1, 1);
    Db = repelem(D(1:(n - 1), :), nNew, 1);
    K0ab = reshape(fk0(A, B, Da, Db), nNew, []);
    k0aa = fk0(XNew, XNew, DNew, DNew);
    ps = sum(K0ab, 2) .* 2 + k0aa;
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
