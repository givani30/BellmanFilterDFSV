function [p] = pdf(y,h,~)
rho         = (1 - exp(-h)) ./ (1 + exp(-h));
rho_term    = 1 - rho.^2;
y_term      = - (y(1)^2 + y(2)^2 - 2 * rho * y(1) * y(2)) ./ (2 * rho_term);
p           = exp(y_term) ./ sqrt(rho_term);
end