function [p] = pdf(y, a, extra_par)
mu          = a;
sigma       = extra_par(1);
nu          = min(max(extra_par(2),2.1),100);
term        = gamma((nu + 1) / 2) / (sqrt(pi * (nu - 2)) * gamma(nu/2) * sigma);
y1          = 1 + (y - mu).^2 ./ (sigma^2 * (nu - 2));
p           = term * y1.^(-(nu + 1) / 2);
end