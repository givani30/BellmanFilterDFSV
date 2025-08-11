function [score] = score(y,a,kappa,~)
lambda  = exp(a);
kappa   = max(min(kappa,40),1);
score   = y - lambda .* (kappa + y) ./ (kappa + lambda);
end

