function k = symdef_k2c(alpha, beta)
% k = symdef_k2c(alpha, beta) returns the symbolic definition of the
% score-based kernel for the two-component Gaussian mixture target.
%
% Input:
% alpha - kernel parameter, e.g. alpha = 1.
% beta  - kernel parameter, e.g. beta = -0.5;
%
% Output:
% k     - symbolic definition of the kernel. The two 1-by-2 vector
%         arguments are named 'a' and 'b'.
%
% Date: January 12, 2018

    y = sym('y', [1, 2], 'real');
    a = sym('a', [1, 2], 'real');
    b = sym('b', [1, 2], 'real');
    d = 1.5;
    m1 = [-d, 0];
    m2 = [d, 0];
    p = log( ...
        exp(-0.5 * (y - m1) * (y - m1)') + ...
        exp(-0.5 * (y - m2) * (y - m2)'));
    u(y) = gradient(p, y)';
    t = u(a(1), a(2)) - u(b(1), b(2));
    k = (alpha + t * t') .^ beta;
end
