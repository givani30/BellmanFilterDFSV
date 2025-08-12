function [output] = info(data,a)
output                = diag(~isnan(data)) .* diag(exp(a));
end

