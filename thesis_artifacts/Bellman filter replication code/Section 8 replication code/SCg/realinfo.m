function [real_info] = realinfo(y,a,~)
% rho         = (1 - exp(-a)) / (1 + exp(-a));
rho         = 2 * (1 ./ (1 + exp(- a))- 0.5);
rho_term    = 1 - rho.^2;
z1          = y(1,:) - rho .* y(2,:);
z2          = y(2,:) - rho .* y(1,:);
real_info   = 0.25 * (z1.^2 + z2.^2) ./ rho_term - 0.25 * rho_term;
end

