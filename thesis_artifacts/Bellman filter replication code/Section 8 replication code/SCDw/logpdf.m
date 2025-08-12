function [logp] = logpdf(y,a,k,~)
beta    = exp(a);
logp    = log(k) + (k - 1) * (log(y) - log(beta)) - log(beta) - (y / beta)^k;
end

