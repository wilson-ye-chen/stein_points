function [Chain, Accept, Scale] = gmrwmetroptune( ...
    kernelfun, start, block, scale0, Sigma, w, s, ...
    targAcc, accTol, nTune, nIter)
% [Chain, Accept, Scale] = gmrwmetroptune(kernelfun, start, block, ...
% scale0, Sigma, w, s, targAcc, accTol, nTune, nIter) samples from a target
% distribution using the Gaussian-mixture random-walk Metropolis algorithm.
% The scale of the covariance matrix of the proposal distribution for each
% block is tuned to obtain the target acceptance rate.
%
% Input:
% kernelfun - handle to the log kernel function of the target density.
% start     - vector of the starting values of the Markov chain.
% block     - cell array where each cell contains a vector of indices of
%             those parameters that form a block.
% scale0    - vector of initial scales of the blocks (e.g., 2.38 / sqrt(d_j),
%             where d_j is the dimension of the j-th block.).
% Sigma     - cell array of covariance matrices of the multivariate Gaussian
%             proposals of the blocks.
% w         - row vector of weights of the mixture components of the proposal
%             distributions (e.g., [0.85, 0.1, 0.05]).
% s         - row vector of scales of the mixture components of the proposal
%             distributions (e.g., [1, 10, 100]).
% targAcc   - vector of target acceptance rates (e.g., 0.44, 0.35, 0.234).
% accTol    - vector of additive acceptance rate tolerances (e.g., 0.075).
% nTune     - number of tuning iterations (e.g., 200).
% nIter     - number of MCMC iterations.
%
% Output:
% Chain     - Markov chain of points.
% Accept    - matrix of indicators for whether a move is accepted, where the
%             rows correspond to MCMC iterations, and the columns correspond
%             to blocks.
% Scale     - trace of tuned scales for each of the blocks.
%
% Author: Wilson Ye Chen <yche5077@uni.sydney.edu.au>
% Date:   February 3, 2015

    % Initialise the chain
    Chain = zeros(nIter, numel(start));
    Chain(1, :) = start;
    
    % Evaluate log-kernel of starting point
    oldKernel = kernelfun(start);
    
    % Initial scales
    nBlock = numel(block);
    Scale = zeros(nIter, nBlock);
    Scale(1, :) = scale0;
    
    % Acceptance indicators
    Accept = zeros(nIter, nBlock);
    
    % For each MCMC iteration
    for i = 2:nIter
        % Copy forward the chain
        Chain(i, :) = Chain((i - 1), :);
        
        % Copy forward the scales
        Scale(i, :) = Scale((i - 1), :);
        
        % For each block
        for j = 1:nBlock
            % Generate a proposal for the block
            subproposal = nmixrnd( ...
                Chain(i, block{j}), ...
                (Scale(i, j) .^ 2) .* Sigma{j}, ...
                w, s);
            
            % Construct the proposal
            proposal = Chain(i, :);
            proposal(block{j}) = subproposal;
            
            % Compute the log acceptance probability
            kernel = kernelfun(proposal);
            accPr = kernel - oldKernel;
            
            % Accept or reject the block
            if accPr >= 0
                Chain(i, :) = proposal;
                oldKernel = kernel;
                Accept(i, j) = 1;
            elseif log(unifrnd(0, 1)) < accPr
                Chain(i, :) = proposal;
                oldKernel = kernel;
                Accept(i, j) = 1;
            end
            
            % Tune the scale for the block
            if ~mod(i, nTune)
                tempAccept = Accept((i - nTune + 1):i, j);
                accRate = sum(tempAccept) ./ nTune;
                if abs(accRate - targAcc(j)) > accTol(j)
                    accRate = min(max(accRate, eps), 1 - eps);
                    Scale(i, j) = ...
                        Scale(i, j) .* ...
                        norminv(targAcc(j) ./ 2, 0, 1) ./ ...
                        norminv(accRate ./ 2, 0, 1);
                end
            end
        end
    end
end

function x = nmixrnd(mu, Sigma, w, s)
% x = nmixrnd(mu, Sigma, w, s) generates a random number
% from a scale-mixture of normal distributions.
%
% Input:
% mu    - row vector of means,
% Sigma - covariance matrix,
% w     - row vector of mixture weights,
% s     - row vector of scales.

    b = mnrnd(1, w);
    x = mvnrnd(mu, ((b * s') .* Sigma));
end
