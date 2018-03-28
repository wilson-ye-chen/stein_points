function pos = getpos(ax, x, y, w, h)
% pos = getpos(ax, x, y, w, h) returns the position vector in figure space.
%
% Input:
% ax   - handle to axes.
% x    - x position in axes space.
% y    - y position in axes space.
% w    - width in axis space.
% h    - height in axis space.
%
% Output:
% pos  - position vector in figure space for [x, y, w, h].
%
% Date: February 9, 2018

    p = get(ax, 'position');
    xl = get(ax, 'xlim');
    yl = get(ax, 'ylim');
    xScale = p(3) ./ (xl(2) - xl(1));
    yScale = p(4) ./ (yl(2) - yl(1));
    xDelta = x - xl(1);
    yDelta = y - yl(1);
    posX = xDelta .* xScale + p(1);
    posY = yDelta .* yScale + p(2);
    posW = w .* xScale;
    posH = h .* yScale;
    pos = [posX, posY, posW, posH];
end
