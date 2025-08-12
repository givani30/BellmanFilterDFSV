function [score] = score(y,a,~)
% rho     = (1 - exp(-h)) / (1 + exp(-h));
rho         = 2 * (1 ./ (1 + exp(- a)) - 0.5);
rho_term    = 1 - rho.^2;
z1          = y(1,:) - rho .* y(2,:);
z2          = y(2,:) - rho .* y(1,:);
score       = 0.5 * rho + 0.5 * z1 .* z2 ./ rho_term;
end

