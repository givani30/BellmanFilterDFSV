function [logp] = logpdf(y,a,~)
lambda  = exp(a);
logp    = y * a - lambda - log(gamma(y+1));
end

