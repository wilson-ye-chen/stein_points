function [Theta, Accept, mapc, ThetaAd, Scale] = mcmc_igarch( ...
    r, nIter, nDiscard)
% [Theta, Accept, mapc, ThetaAd, Scale] = mcmc_igarch(r, nIter, nDiscard)
% implements an adaptive random walk Metropolis algorithm to generate samples
% from the posterior of the IGARCH model with Gaussian innovations.
%
% Date: January 30, 2018

    % Log-posterior
    h1 = var(r);
    fp = @(t)fp_igarch([t(1), t(2)], r, h1) + prior(t);

    % Initial parameter values
    nDim = 2;
    theta0 = [0.02, 0.1];

    % Define parameter blocking
    block{1} = 1:nDim;

    % Block dependent configuration
    SigProp0{1} = eye(nDim);
    scale0 = 2.38 ./ sqrt(nDim);
    targAcc = targaccrate(nDim);
    accTol = 0.075;

    % Global configuration
    w = [0.7, 0.15, 0.15];
    s = [1, 100, 0.01];
    nTune = 200;
    nIterAd = 12000;
    nDiscardAd = 2000;
    minAdapt = 2;
    maxAdapt = 30;
    mapcTol = 0.1;

    % Run the adaptive random-walk Metropolis sampler
    [ThetaAd, Scale, SigProp, Accept, mapc] = gmrwmetropadapt( ...
        fp, theta0, block, scale0, SigProp0, w, s, ...
        targAcc, accTol, nTune, nIterAd, nDiscardAd, ...
        minAdapt, maxAdapt, mapcTol);

    % Stop adapting after tuning period
    theta0 = mean(ThetaAd, 1);
    aveScl = mean(Scale, 1);
    SigProp{1} = (aveScl .^ 2) .* SigProp{1};
    [Theta, Accept] = gmrwmetrop( ...
        fp, theta0, block, SigProp, w, s, nIter);

    % Discard a few initial samples
    Theta = Theta((nDiscard + 1):end, :);
    Accept = Accept((nDiscard + 1):end, :);
end

function logPrr = prior(theta)
    if theta(1) > 0 && theta(2) > 0 && theta(2) < 1
        logPrr = 0;
    else
        logPrr = -inf;
    end
end

function accRate = targaccrate(nDim)
    if nDim == 1
        accRate = 0.44;
    elseif nDim >= 2 && nDim <= 4
        accRate = 0.35;
    else
        accRate = 0.234;
    end
end
