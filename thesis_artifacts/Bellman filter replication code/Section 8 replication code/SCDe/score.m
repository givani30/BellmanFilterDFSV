function [score] = score(y,a,~)
lambda      = exp(a);
score       = 1 - lambda .* y;
end

