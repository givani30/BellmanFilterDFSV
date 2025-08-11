function [logp] = logpdf(y,a,~)
sigma2      = exp(a);
% logp        = - 0.5 * (log(2 * pi) + a + (y - mu).^2 / sigma2);
% Equivalent
logp        = - 0.5 * (a + y^2 / sigma2);
end

