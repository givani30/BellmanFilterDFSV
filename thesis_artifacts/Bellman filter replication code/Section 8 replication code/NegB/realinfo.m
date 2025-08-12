function [real_info] = realinfo(y,a,kappa,~)
lambda      = exp(a);
kappa       = max(min(kappa,40),1);
real_info   = kappa .* lambda .* (kappa + y) ./ ((kappa + lambda).^2);
end

