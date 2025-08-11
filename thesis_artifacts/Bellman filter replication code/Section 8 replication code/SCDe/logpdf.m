function [logp] = logpdf(y,a,~)
lambda  = exp(a);
logp    = a - lambda * y;
end

