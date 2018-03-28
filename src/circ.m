function X = circ(c, r, n)
% X = circ(c, r, n) generates a set of equally spaced points on a circle.
%
% Date: May 20, 2017

    t = linspace(-pi, pi, n + 1);
    t = t(1:(end - 1))';
    x1 = r .* cos(t) + c(1);
    x2 = r .* sin(t) + c(2);
    X = [x1, x2];
end
