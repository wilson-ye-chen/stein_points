function [Mu, S, w] = gmparam_2c()
% [Mu, S, w] = gmparam_2c() returns the parameter values for a two-
% component bivariate Gaussian mixture distribution.
%
% Output:
% Mu - 2-by-2 matrix of mean vectors.
% S  - 2-by-2-by-2 array of covariance matrices.
% w  - 2-by-1 vector of weights.
%
% Date: January 11, 2018

    d = 1.5;
    Mu = [-d, 0; d, 0];
    S = repmat(eye(2), 1, 1, 2);
    w = [0.5; 0.5];
end
