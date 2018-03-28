%%
% File: gmparam_rnd_test.m
% Purpose:
% Creates a sequence of randomly generated Gaussian mixture distributions
% and plots the density contours one by one.
% Date: January 9, 2018
%%

% Default seed sequence
if ~exist('seed')
    seed = 0:500;
end

% Write to a video file
v = VideoWriter('gmparam_rnd_out.avi');
v.FrameRate = 2;
v.Quality = 50;
open(v);

% Iterate over seed values
for i = 1:numel(seed)
    % Fix seed
    rng(seed(i));

    % Generate random parameter values
    [Mu, S, w] = gmparam_rnd();
    fp = @(X)fp_gaussmix(X, Mu, S, w);

    % Evaluate the density over a grid
    lb = [-10, -10];
    ub = [10, 10];
    nStep = 100;
    st1 = linspace(lb(1), ub(1), nStep);
    st2 = linspace(lb(2), ub(2), nStep);
    [T1, T2] = meshgrid(st1, st2);
    T = [T1(:), T2(:)];
    Z = reshape(fp(T), nStep, nStep);

    % Plot the contours
    contour(T1, T2, Z, 'levelstep', 0.001, 'linewidth', 1);
    hold on;
    plot(Mu(:, 1), Mu(:, 2), '+r', 'linewidth', 1);
    hold off;
    title(sprintf('Random GM - %d', seed(i)));

    % Add frame
    frame = getframe(gcf);
    writeVideo(v, frame);
end
close(v);
