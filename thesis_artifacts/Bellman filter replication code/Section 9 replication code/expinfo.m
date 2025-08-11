function [output] = expinfo(data,a,~)
output    = exp(a);
output(isnan(output)) = 0; % if the output is nan, set to zero
end

