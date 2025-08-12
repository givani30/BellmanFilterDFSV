function [logp] = logpdf(y,a,~)
rho         = (1 - exp(-a)) / (1 + exp(-a));
rho_term    = 1 - rho^2;
y_term      = - (y(1)^2 + y(2)^2 - 2 * rho * y(1) * y(2)) / (2 * rho_term);
logp        = y_term - 0.5 * log(rho_term);% - log(2 * pi) term omitted;
end

