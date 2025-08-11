function [p] = pdf(y,a,nu,~)
rho         = (1 - exp(-a)) ./ (1 + exp(-a));
rho_term    = 1 - rho.^2;
nu          = min(max(nu,4),40);
y_term      = 1 + (y(1)^2 + y(2)^2 - 2 * rho * y(1) * y(2)) ./ ((nu-2) * rho_term);
p           = nu/(2*pi)/(nu-2) ./ sqrt(rho_term) .* y_term.^(- (nu + 2) / 2) ;
end