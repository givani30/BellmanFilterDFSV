function [real_info] = realinfo(y,a,~,~)
beta        = exp(a);
real_info   = y ./ beta;
end

