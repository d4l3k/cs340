function [y] = myMode(x)
if isempty(x)
	y = NaN;
else
	y = mode(x);
end