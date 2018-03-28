function p = fp_garch(Phi, y, h1)
% p = fp_garch(Phi, y, h1) evaluates the un-normalised log-posterior
% density of the posterior of the zero-mean Gaussian-GARCH(1,1) model.
%
% Input:
% Phi - n-by-3 matrix of n sets of log GARCH parameters.
% y   - vector of observations.
% h1  - scalar of initial variance.
%
% Output:
% D   - n-by-1 vector of log-posterior values.
%
% Date: October 26, 2017

    n = size(Phi, 1);
    nObs = numel(y);
    Tht = exp(Phi);
    tht1 = Tht(:, 1);
    tht2 = Tht(:, 2);
    tht3 = Tht(:, 3);

    h = ones(n, 1) .* h1;
    p = zeros(n, 1);
    for i = 2:nObs
        h = tht1 + tht2 .* y(i - 1) .^ 2 + tht3 .* h;
        p = p + log(normpdf(y(i), 0, sqrt(h)));
    end
end
