function [real_info] = realinfo(y,a,~)
lambda      = exp(a);
real_info   = lambda .* y;
end

