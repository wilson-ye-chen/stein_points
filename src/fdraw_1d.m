function fdraw_1d(f, xNew, l, u, n)
% fdraw_1d(f, xNew, l, u, n)

    x = linspace(l, u, n);
    plot(x, f(x), '-k', 'linewidth', 1.5);
    hold on;
    plot(xNew, f(xNew), 'r.', 'markersize', 25);
    hold off;
    input('Press <Enter> to continue...');
end
