function [logp] = logpdf(y,a,k,~)
beta    = exp(a);
k       = max(k,1);
logp    = (k-1) * log(y) - y / beta - log(gamma(k)) - k * log(beta);
end

