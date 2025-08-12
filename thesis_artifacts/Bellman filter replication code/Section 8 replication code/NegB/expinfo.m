function [exp_info] = expinfo(~,a,kappa,~)
lambda     = exp(a);
kappa      = max(min(kappa,40),1);
exp_info   = kappa .* lambda ./ (kappa + lambda);
end

