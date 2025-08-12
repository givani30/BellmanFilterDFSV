function [p] = pdf(y,a,~)
sigma2  = exp(a);
% p       = 1 ./ sqrt(2 * pi * sigma2) .* exp(- 0.5 ./ sigma2 * y^2);
%Equivalent
p       = 1 ./ sqrt(sigma2) .* exp(- 0.5 ./ sigma2 * y^2);

end