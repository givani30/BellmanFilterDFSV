function [score] = score(y,a,nu,~)
% rho    = (1 - exp(-a)) / (1 + exp(-a));
rho     = 2 * (1 ./ (1 + exp(- a))- 0.5);
z1      = y(1,:) - rho .* y(2,:);
z2      = y(2,:) - rho .* y(1,:);
nu      = min(max(nu,4),40);
wt      = (nu + 2) ./ (nu - 2 + (y(1,:).^2 + y(2,:).^2 - 2 .* rho .* y(1,:) .* y(2,:)) ./ (1 - rho.^2) );
score   = 0.5 * rho + 0.5 * wt .* z1 .* z2 ./ (1 - rho.^2);
end

