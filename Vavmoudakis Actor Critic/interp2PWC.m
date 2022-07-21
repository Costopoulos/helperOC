% interp2PWC is a function that approximates the input vector data to a
% piecewise continuous function given the initial and final time provided

function [f] = interp2PWC(y,xi,xf)
[m,~] = size(y); % get size of the input vec
x = linspace(xi,xf,m); %create x vector corresponding to y
[c,ia,~] = unique(y,'stable'); % figure out who and where are the unique points
f = mkpp([xi x(ia(2:end)') xf],c); % interpolate in a piecewise continuous fcn with corresponding time vec
end
