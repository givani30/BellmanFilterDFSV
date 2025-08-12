function [output] = logpdf2(y,a,nu)
rho        = (1-exp(-a)) ./ (1+exp(-a));
Sigma      = [ 1 , rho ; rho , 1 ];
detSigma   = 1 - rho^2;
invSigma   = 1/detSigma * [ 1 , -rho ; -rho , 1 ];
%output1   = gamma(nu/2+1) / gamma(nu/2) / (nu-2) / pi / sqrt(detSigma)  * (1+1/(nu-2)* y'*invSigma*y )^(-nu/2-1); 
output2    = nu/2                        / (nu-2) / pi / sqrt(detSigma)  * (1+1/(nu-2)* y'*invSigma*y )^(-nu/2-1); 
output     = log(output2);
end