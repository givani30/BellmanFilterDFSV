function [p] = pdf(y, a, nu)
sigma2      = exp(a);
sigma       = sigma2.^0.5;
nu          = min(max(nu,4),40);
y_term      = 1 + (y^2) ./ ((nu - 2) * sigma2);
% p           = gamma((nu + 1) / 2) ./ (sqrt(pi * (nu - 2)) * gamma(nu / 2) * sigma) .* (y_term).^(- (nu + 1) / 2);
% Equivalent
p           = gamma((nu + 1) / 2) ./ (sqrt((nu - 2)) * gamma(nu / 2) * sigma) .* (y_term).^(- (nu + 1) / 2);
end