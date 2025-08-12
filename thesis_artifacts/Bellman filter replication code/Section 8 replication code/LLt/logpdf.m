function [logp] = logpdf(y,a,extra_par)
mu          = a;
sigma       = extra_par(1);
nu          = min(max(extra_par(2),2.1),100);
y1          = 1 + (y - mu)^2 / (sigma^2 * (nu - 2));
logp        = log(gamma((nu + 1) / 2)) - (0.5 * log(nu - 2) + log(gamma(nu/2)) + log(sigma)) -(nu+1)/2 * log(y1);
end

