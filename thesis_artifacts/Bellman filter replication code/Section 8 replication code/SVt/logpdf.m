function [logp] = logpdf(y,a,nu)
mu          = 0;
sigma2      = exp(a);
%sigma      = sigma2^.5;
nu          = min(max(nu,4),40);
y2          = 1 + (y^2)./((nu-2)*sigma2);
if gamma(nu) == Inf
    logp = - 0.5 * (log(2 * pi) + a + (y - mu).^2 / sigma2);
else
    logp    = log(gamma((nu+1)/2))-log(gamma(nu/2))-0.5*log(nu-2)-0.5*log(pi)...
        -0.5*a-((nu+1)/2)*log(y2);
end
end

