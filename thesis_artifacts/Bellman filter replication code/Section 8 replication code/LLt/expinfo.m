function [expinfo] = expinfo(~,a,extra_par)
sigma       = max(extra_par(1),0.001);
nu          = min(max(extra_par(2),2.1),100);
expinfo     = ones(size(a)) * nu * (nu + 1) / sigma^2 / (nu-2) / (nu+3);
end

