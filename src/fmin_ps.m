function [xMin, dMin, fMin, nEval] = fmin_ps(f, nDim, nPart, lb, ub)
% [xMin, dMin, fMin, nEval] = fmin_ps(f, nDim, nPart, lb, ub) minimises
% the objective function using a particle-swarm optimisation algorithm.
%
% Input:
% f     - function handle to the objective. This function must return
%         the score matrix of the target density as its second output.
% nDim  - number of objective dimensions.
% nPart - swarm size.
% lb    - 2-vector of lower bounds of the search grid.
% ub    - 2-vector of upper bounds of the search grid.
%
% Output:
% xMin  - 1-by-2 vector that minimises f.
% dMin  - 1-by-2 score vecotr at xMin.
% fMin  - objective value at xMin.
% nEval - number of function evaluations.
%
% Date: October 28, 2017

    opt = optimoptions('particleswarm', ...
        'swarmsize', nPart, ...
        'usevectorized', true, ...
        'display', 'off');
    [xMin, fMin, ~, out] = particleswarm(f, nDim, lb, ub, opt);
    nEval = out.funccount;

    % One last call to the objective to obtain the score vector
    [~, dMin] = f(xMin);
    nEval = nEval + 1;
end
