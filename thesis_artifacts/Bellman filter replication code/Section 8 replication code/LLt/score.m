function [score] = score(y,a,extra_par)
mu          = a;
sigma       = extra_par(1);
nu          = min(max(extra_par(2),2.1),100);
et          = (y - mu) ./ sigma;
score       = 1 ./ sigma * (nu + 1) .* et ./ (nu - 2 + et.^2);
end

