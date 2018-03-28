function [Chain, Accept] = gmrwmetrop( ...
    kernelfun, start, block, Sigma, w, s, nIter)
% [Chain, Accept] = gmrwmetrop(kernelfun, start, block, Sigma, w, s, nIter)
% samples from a target distribution using the Gaussian-mixture random-walk
% Metropolis algorithm.
%
% Input:
% kernelfun - handle to the log kernel function of the target density.
% start     - vector of the starting values of the Markov chain.
% block     - cell array where each cell contains a vector of indices of
%             those parameters that form a block.
% Sigma     - cell array of covariance matrices that are to be scaled to
%             construct the multivariate Gaussian-mixture proposals of the
%             blocks.
% w         - row vector of weights of the mixture components of the proposal
%             distributions (e.g., [0.85, 0.1, 0.05]).
% s         - row vector of scales of the mixture components of the proposal
%             distributions (e.g., [1, 10, 100]).
% nIter     - number of MCMC iterations.
%
% Output:
% Chain     - Markov chain of points.
% Accept    - matrix of indicators for whether a move is accepted, where the
%             rows correspond to MCMC iterations, and the columns correspond
%             to blocks.
%
% Author: Wilson Ye Chen <yche5077@uni.sydney.edu.au>
% Date:   February 4, 2015

    % Initialise the chain
    Chain = zeros(nIter, numel(start));
    Chain(1, :) = start;
    
    % Evaluate log-kernel of starting point
    oldKernel = kernelfun(start);
    
    % Acceptance indicators
    nBlock = numel(block);
    Accept = zeros(nIter, nBlock);
    
    % For each MCMC iteration
    for i = 2:nIter
        % Copy forward the chain
        Chain(i, :) = Chain((i - 1), :);
        
        % For each block
        for j = 1:nBlock
            % Generate a proposal for the block
            subproposal = nmixrnd( ...
                Chain(i, block{j}), Sigma{j}, w, s);
            
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
