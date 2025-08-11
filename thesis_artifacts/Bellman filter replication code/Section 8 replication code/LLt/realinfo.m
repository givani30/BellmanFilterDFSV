function [real_info] = realinfo(y,a,extra_par)
mu          = a;
sigma       = extra_par(1);
nu          = min(max(extra_par(2),2.1),100);
et          = (y - mu) ./ sigma;
real_info   = (nu + 1) / sigma^2 * (nu - 2 - et.^2) ./ (nu - 2 + et.^2).^2;
end

