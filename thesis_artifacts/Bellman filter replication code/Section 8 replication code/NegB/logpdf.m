function [logp] = logpdf(y,h,kappa,~)
lambda  = exp(h);
kappa   = max(min(kappa,40),1);
logp    = log(gamma(kappa + y)) - log(gamma(kappa)) - log(gamma(1 + y)) ...
    + kappa * (log(kappa) - log(kappa + lambda)) + y * (h - log((kappa + lambda)));
end

