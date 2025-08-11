function [p] = pdf(y,a,k,~)
beta  = exp(a);
k     = max(k,1);
p     = y^(k - 1) * exp(-y ./ beta) ./ (gamma(k) * beta.^k);
end