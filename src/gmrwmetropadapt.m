function [Chain, Scale, Sigma, Accept, mapc] = gmrwmetropadapt( ...
    kernelfun, start0, block, scale0, Sigma0, w, s, ...
    targAcc, accTol, nTune, nIter, nDiscard, ...
    minAdapt, maxAdapt, mapcTol)
% [Chain, Scale, Sigma, Accept, mapc] = gmrwmetropadapt(kernelfun, ...
% start0, block, scale0, Sigma0, w, s, targAcc, accTol, nTune, nIter, ...
% nDiscard, minAdapt, maxAdapt, mapcTol) samples from a target distribution
% using an adaptive Gaussian-mixture random-walk Metropolis algorithm.
%
% Input:
% kernelfun - handle to the log kernel function of the target density.
% start0    - vector of the starting values of the Markov chain.
% block     - cell array where each cell contains a vector of indices of
%             those parameters that form a block.
% scale0    - vector of initial scales of the blocks (e.g., 2.38 / sqrt(d_j),
%             where d_j is the dimension of the j-th block.).
% Sigma0    - cell array of initial covariance matrices of the multivariate
%             Gaussian proposals of the blocks.
% w         - row vector of weights of the mixture components of the proposal
%             distributions (e.g., [0.85, 0.1, 0.05]).
% s         - row vector of scales of the mixture components of the proposal
%             distributions (e.g., [1, 10, 100]).
% targAcc   - vector of target acceptance rates (e.g., 0.44, 0.35, 0.234).
% accTol    - vector of additive acceptance rate tolerances (e.g., 0.075).
% nTune     - number of tuning iterations (e.g., 200).
% nIter     - number of MCMC iterations per adaptive iteration.
% nDiscard  - number of discarded initial MCMC iterations when updating the
%             covariance matrices.
% minAdapt  - minimum number of adaptive iterations (e.g., 1).
% maxAdapt  - maximum number of adaptive iterations (e.g., 30).
% mapcTol   - the algorithm terminates when the mean absolute percentage
%             change (MAPC) of the marginal standard deviations from the
%             previous adaptive iteration is less than mapcTol (e.g., 0.1).
%
% Output:
% Chain     - Markov chain of points.
% Scale     - trace of tuned scales of the blocks.
% Sigma     - cell array of sample covariance matrices of the blocks.
% Accept    - matrix of indicators for whether a move is accepted, where the
%             rows correspond to MCMC iterations, and the columns correspond
%             to blocks.
% mapc      - vector of MAPCs of all adaptive iterations.
%
% Author: Wilson Ye Chen <yche5077@uni.sydney.edu.au>
% Date:   December 1, 2015

    % Number of parameter blocks
    nBlock = numel(block);
    
    % Initial states
    start = start0;
    Sigma = Sigma0;
    sdOld = 0;
    mapc = zeros(maxAdapt, 1);
    i = 0;
    stop = (maxAdapt == 0);
    
    % Adaptive iterations
    while ~stop
        % Run random-walk sampler
        [Chain, Accept, Scale] = gmrwmetroptune( ...
            kernelfun, start, block, scale0, Sigma, w, s, ...
            targAcc, accTol, nTune, nIter);
        
        % Update starting values
        start = Chain(end, :);
        
        % Update covariance matrices
        for j = 1:nBlock
            S = cov(Chain((nDiscard + 1):end, block{j}));
            [~, notPd] = chol(S);
            if notPd
                warning('Covariance update failed!');
            else
                Sigma{j} = S;
            end
        end
        
        % Increment adaptive iteration count
        i = i + 1;
        
        % Convergence is measured by the change in marginal standard
        % deviations of the chain
        sd = std(Chain);
        mapc(i) = mean(abs((sd - sdOld) ./ sdOld));
        sdOld = sd;
        
        % Be verbose
        disp(['Epoch ', num2str(i), ': MAPC = ', num2str(mapc(i))]);
        
        % Stopping conditions
        if (i >= minAdapt && mapc(i) <= mapcTol) || i >= maxAdapt
            stop = true;
        end
    end
    
    % Trim output matrices
    Chain = Chain((nDiscard + 1):end, :);
    Scale = Scale((nDiscard + 1):end, :);
    Accept = Accept((nDiscard + 1):end, :);
    mapc = mapc(1:i);
end
