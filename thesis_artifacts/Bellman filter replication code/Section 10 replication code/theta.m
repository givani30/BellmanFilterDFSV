function [output] = theta(lag,rho_vector)
% covariance between eta_t and eta_{t-lag}
if lag > length(rho_vector)-1
    output = 0;
elseif lag == 0;
    output = 1;
else
    output = sum(rho_vector(1:end-lag) .* rho_vector(1+lag:end));
end

