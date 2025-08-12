function [exp_info] = expinfo(~,a,nu)
nu          = min(max(nu,4),40);
exp_info    = nu/(2*nu+6) * ones(size(a));
end

