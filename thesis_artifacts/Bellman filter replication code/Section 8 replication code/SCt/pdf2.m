function [p] = pdf2(y,a,nu,~)
rho         = (1 - exp(-a)) ./ (1 + exp(-a));
rho_term    = 1 - rho.^2;
gamma_term  = gamma((nu + 2) / 2) * gamma(nu/2) / gamma((nu + 1) / 2)^2 ;
y_term      = 1 + (y(1)^2 + y(2)^2 - 2 * rho * y(1) * y(2)) ./ (nu * rho_term);
product     = (1 + y(1)/nu)^(- (nu + 1)/2) * (1 + y(2)/nu)^(- (nu + 1)/2);
p           = gamma_term ./ (sqrt(rho_term) * product) .* y_term.^(- (nu + 2) / 2);
end