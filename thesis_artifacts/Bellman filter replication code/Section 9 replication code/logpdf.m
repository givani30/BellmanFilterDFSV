function [output] = logpdf(data,alpha)
% select observations that are available
y           = data;
y(isnan(y)) = 0; % transformed version of the data where missing values are zero
% compute the log-factorial term
logfactorial = log(factorial(y));
% in case any log factorial is infinite, use Strirling's formula instead
if sum(logfactorial)==inf
indices               = isinf(logfactorial);
problematic_y         = y(indices);
logfactorial(indices) = problematic_y .* log(problematic_y) - problematic_y;
end
% compute output as one would normally
output = y.*alpha-logfactorial-exp(alpha); %this output contains no nan's anymore, but it is "wrong" when y used to be nan
% so we set output to zero if the original data was nan
output = (~isnan(data)) .* output;
% sum over all observations
output = sum(output,1);
end
