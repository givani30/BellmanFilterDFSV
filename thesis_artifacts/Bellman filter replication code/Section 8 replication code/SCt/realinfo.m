function [real_info] = realinfo(y,a,nu,~)
% rho       = (1 - exp(-a)) / (1 + exp(-a));
rho         = 2 * (1 ./ (1 + exp(- a))- 0.5);
z1          = y(1,:) - rho .* y(2,:);
z2          = y(2,:) - rho .* y(1,:);
nu          = min(max(nu,4),40);
wt          = (nu + 2) ./ (nu - 2 + (y(1,:).^2 + y(2,:).^2 - 2 * rho .* y(1,:) .* y(2,:)) ./  (1 - rho.^2));
real_info   = 0.25 * wt .* (z1.^2 + z2.^2) ./ (1 - rho.^2) - 0.25 * (1 - rho.^2) ...
    - 0.5 * wt.^2 ./ (nu + 2) .* (z1.^2 .* z2.^2) ./ (1 - rho.^2).^2;
end

