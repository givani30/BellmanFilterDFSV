function [output] = score(data,a)
output              = data - exp(a);
output(isnan(data)) = 0; % if the data is nan, set to zero
end

