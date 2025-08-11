function [p] = pdf(y,a,k,~)
beta    = exp(a);
p       = k * (y ./ beta).^(k - 1) ./ (beta .* exp((y ./ beta).^k));
end