function [exp_info] = expinfo(~,a,nu,~)
% rho       = (1 - exp(-a)) / (1 + exp(-a));
rho         = 2 * (1 ./ (1 + exp(- a))- 0.5);
nu          = min(max(nu,4),40);
exp_info    = 0.25 * (2 + nu * ( 1 + rho.^2 ) ) ./ ( nu + 4 );
end

