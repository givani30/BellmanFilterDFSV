function [real_info] = realinfo(y,a,k,~)
beta        = exp(a);
real_info   = k^2 * (y ./ beta).^k;
end

