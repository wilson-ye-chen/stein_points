function [D, p] = fscr_gp(X, y, t, sigma, draw)
% [D, p] = fscr_gp(X, y, t, sigma, draw) returns scores for a particular
% posterior. This originates from an application of Gaussian process
% regression and the posterior distribution is defined on the kernel
% parameters given the observed dataset.
%
% Input:
% X     - n-by-nDim matrix of points.
% y     - n-by-1 column vector of dependent variable values.
% t     - n-by-1 column vector of independent variable values.
% sigma - observation noise parameter.
% draw  - logical indicator of whether to produce a plot.
%
% Output:
% D     - n-by-nDim matrix of scores.
% p     - handle to an (un-normalised) pdf of the target (optional).
%
% Date: January 30, 2018

    n = size(X, 1);
    nObs = numel(y);

    % Distance matrix
    Dist_00 = pdist2(t, t);
    % Covariance matrix - deterministic part
    C_00 = @(x)exp(x(1)) .* exp(-exp(x(2)) .* Dist_00 .^ 2);
    % Measurement noise model
    N_00 = @(x)sigma .^ 2 .* eye(nObs);

    % Each X(i, :) corresponds to a regression model. Here we plot all
    % n of these fitted models.
    if draw
        m = 100;
        t_grid = linspace(min(t), max(t), m)';
        Dist_10 = pdist2(t_grid, t);
        % Cross-covariance matrix
        C_10 = @(x)exp(x(1)) .* exp(-exp(x(2)) .* Dist_10 .^ 2);
        % Regression function evaluated on grid
        mu = @(x)C_10(x) * ((C_00(x) + N_00(x)) \ y);

        figure();
        plot(t, y, 'rx', 'linewidth', 1.2);
        hold on;
        xlabel('t');
        ylabel('y');
        title('Regression Problem');
        for i = 1:n
            x = X(i, :);
            plot(t_grid, mu(x), 'color', [0, 0, 0, 0.3]);
        end
    end

    % Return (un-normalised) density
    if nargout > 1
        p = @(X)fp(X, y, C_00, N_00);
    end

    % Compute scores (Eqn. 5.9 of Rasmissen and Williams, 2006)
    dC1 = @(x)C_00(x);
    dC2 = @(x)C_00(x) .* -exp(x(2)) .* Dist_00 .^ 2;
    D = zeros(n, 2);
    for i = 1:n
        x = X(i, :);
        Cx = C_00(x) + N_00(x);
        dC1x = dC1(x);
        dC2x = dC2(x);
        a = Cx \ y;
        D(i, 1) = ...
            0.5 .* trace(a * (a' * dC1x) - Cx \ dC1x) - ...
            2 .* x(1) ./ (1 + x(1) .^ 2);
        D(i, 2) = ...
            0.5 .* trace(a * (a' * dC2x) - Cx \ dC2x) - ...
            2 .* x(2) ./ (1 + x(2) .^ 2);
    end
end

function p = fp(X, y, C_00, N_00)
    n = size(X, 1);
    nObs = numel(y);
    p = zeros(n, 1);
    for i = 1:n
        x = X(i, :);
        S = C_00(x) + N_00(x);
        [~, notPd] = chol(S, 'upper');
        if notPd
            p(i) = 0;
        else
            p(i) = ...
                mvnpdf(y', zeros(1, nObs), S) .* ...
                tpdf(x(1), 1) .* tpdf(x(2), 1);
        end
    end
end
