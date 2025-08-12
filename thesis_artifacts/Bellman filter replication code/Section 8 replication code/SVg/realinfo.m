function [real_info] = realinfo(y,a,~)
real_info = y.^2 ./ (2 * exp(a));
end

