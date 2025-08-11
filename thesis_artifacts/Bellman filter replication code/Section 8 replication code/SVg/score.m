function [score] = score(y,a,~)
score = y.^2 ./ (2 * exp(a)) - 0.5;
end

