function [score] = score(y,a,nu)
sigma2      = exp(a);
%sigma      = sigma2.^.5;
nu          = min(max(nu,4),40);
wt          = (nu + 1) ./ (nu - 2 + y.^2 ./ sigma2);
score       = wt .* y.^2 ./ (2 * sigma2) - 0.5;
end

