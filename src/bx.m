function X = bx(c, w, n)
% X = bx(c, w, n) generates a set of equally spaced points over a box.
%
% Date: May 20, 2017

    r = w ./ 2;
    v1 = linspace(c(1) - r, c(1) + r, sqrt(n));
    v2 = linspace(c(2) - r, c(2) + r, sqrt(n));
    [X1, X2] = meshgrid(v1, v2);
    X = [X1(:), X2(:)];
end
