function [exp_info] = expinfo(~,a,k,~)
k         = max(k,1);
exp_info  = k * ones(size(a));
end

