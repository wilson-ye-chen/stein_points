function D = fscr_garch(Phi, y, h1)
% D = fscr_garch(Phi, y, h1) evaluates the score function of the posterior
% of the zero-mean Gaussian-GARCH(1,1) model.
%
% Input:
% Phi - n-by-3 matrix of n sets of log GARCH parameters.
% y   - vector of observations.
% h1  - scalar of initial variance.
%
% Output:
% D   - n-by-3 matrix of score function values.
%
% Date: October 26, 2017

    n = size(Phi, 1);
    nObs = numel(y);
    Tht = exp(Phi);
    tht1 = Tht(:, 1);
    tht2 = Tht(:, 2);
    tht3 = Tht(:, 3);

    dh_phi1 = zeros(n, 1);
    dh_phi2 = zeros(n, 1);
    dh_phi3 = zeros(n, 1);
    h = ones(n, 1) .* h1;
    dp_phi1 = zeros(n, 1);
    dp_phi2 = zeros(n, 1);
    dp_phi3 = zeros(n, 1);
    ySq = y .^ 2;
    for i = 2:nObs
        dh_phi1 = tht1 + tht3 .* dh_phi1;
        dh_phi2 = tht2 .* ySq(i - 1) + tht3 .* dh_phi2;
        dh_phi3 = tht3 .* h + tht3 .* dh_phi3;
        h = tht1 + tht2 .* ySq(i - 1) + tht3 .* h;
        dp_h = -1 ./ (2 .* h) + ySq(i) ./ (2 .* h .^ 2);
        dp_phi1 = dp_phi1 + dp_h .* dh_phi1;
        dp_phi2 = dp_phi2 + dp_h .* dh_phi2;
        dp_phi3 = dp_phi3 + dp_h .* dh_phi3;
    end
    D = [dp_phi1, dp_phi2, dp_phi3];
end
