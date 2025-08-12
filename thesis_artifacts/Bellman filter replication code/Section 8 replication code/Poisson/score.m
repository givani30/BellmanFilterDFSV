function [score] = score(y,a,~)
lambda  = exp(a);
score   = y - lambda;
end

