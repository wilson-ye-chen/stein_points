%%
% File: genpts_2c.m
% Purpose:
% Generate point-set plots from the output file of genemd_2c.m.
% Date: February 4, 2018
%%

% Load the output file
load('genemd_2c_100.mat');

alg = {'0', '1', '2', '3', '4', '5'};
lab = { ...
    'mc', ...
    'stein_greedy', ...
    'stein_fw', ...
    'stein_codescent', ...
    'med_greedy', ...
    'svgd'};

for i = 1:numel(alg)
    % Plot contour
    figure();
    t1 = linspace(lb(1), ub(1), nStep)';
    t2 = linspace(lb(2), ub(2), nStep)';
    T = [repelem(t1, nStep), repmat(t2, nStep, 1)];
    p = fp(T);
    Z = reshape(p, nStep, nStep);
    contour(t1, t2, Z, 'levelstep', 0.005, 'linewidth', 0.8);

    % Plot points
    hold on;
    X = eval(['X_', alg{i}]);
    X = X((end - nPart + 1):end, :);
    plot(X(:, 1), X(:, 2), '+r', 'linewidth', 1.2);
    axis([lb(1), ub(1), lb(2), ub(2)]);
    set(gca, 'fontsize', 12);
    xlabel('x_1', 'fontsize', 12);
    ylabel('x_2', 'fontsize', 12);

    % Window setting
    set(gcf, 'renderer', 'painters');
    set(gcf, 'units', 'centimeters');
    set(gcf, 'position', [0.5, 1.5, 7, 7]);

    % Print setting
    set(gcf, 'paperunits', 'centimeters');
    set(gcf, 'paperpositionmode', 'manual');
    set(gcf, 'paperposition', [0, 0, 7, 7]);
    set(gcf, 'papertype', '<custom>');
    set(gcf, 'papersize', [7, 7]);

    % Print to PDF
    name = sprintf('genpts_2c_%s_%d', lab{i}, nPart);
    print(name, '-dpdf');
end
