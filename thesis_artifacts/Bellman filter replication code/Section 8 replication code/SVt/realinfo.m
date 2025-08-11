function [real_info] = realinfo(y,a,nu)
sigma2      = exp(a);
nu          = min(max(nu,4),40);
%sigma      = sigma2.^(1/2);
wt          = (nu + 1) ./ (nu - 2 + y.^2 ./ sigma2);
real_info   = (nu-2)/(nu+1) * wt.^2 .* y.^2 ./ (2 * sigma2);
end

