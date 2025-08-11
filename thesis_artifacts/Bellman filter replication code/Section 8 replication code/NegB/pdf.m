function [p] = pdf(y,h,kappa,~)
lambda  = exp(h);
kappa   = max(min(kappa,40),1);
p       = gamma(kappa + y) / (gamma(kappa) * gamma(1 + y)) * (kappa ./ (kappa + lambda)).^kappa .* ...
            (lambda ./ (kappa + lambda)).^y;
end