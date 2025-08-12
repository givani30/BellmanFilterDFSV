function [logp] = logpdf(y,a,nu,~)
rho         = (1 - exp(-a)) ./ (1 + exp(-a));
rho_term    = 1 - rho.^2;
nu          = min(max(nu,4),40);
y_term      = 1 + (y(1)^2 + y(2)^2 - 2 * rho * y(1) * y(2)) ./ ((nu-2) * rho_term);
logp        = log(nu) - log(2*pi*(nu-2)) -  0.5 * log(rho_term) - 0.5 * (nu + 2) * log(y_term) ;
end

