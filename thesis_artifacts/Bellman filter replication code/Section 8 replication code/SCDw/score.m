function [score] = score(y,a,k,~)
beta        = exp(a);
score       = k * (y ./ beta).^k - k;
end

