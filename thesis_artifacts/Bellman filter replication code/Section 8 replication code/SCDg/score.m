function [score] = score(y,a,k,~)
beta        = exp(a);
k           = max(k,1);
score       = y ./ beta - k;
end

