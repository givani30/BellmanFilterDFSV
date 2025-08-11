function [output] = logpdf(y_t_with_lags,h_t_with_lags,psi)
% This function is meant to give the observation density of y_t, conditional
% on h_t (and its lags) and lags of y_t itself, as in eq (3) of Catania (2021)
% The only difference is that we don't have beta, but we have a constant in
% the dynamic equation.

%% Extract parameter values
mu               = psi(1);
c                = psi(2);
phi              = psi(3);
phi              = max(min(phi,0.999),-.999);
sigma_eta        = abs(psi(4));
rho_vector       = psi(5:end);
rho_vector       = reshape( rho_vector , length(rho_vector) , 1); % make sure its a column vector
rho_0            = rho_vector(1);
no_lags          = length(rho_vector)-1;
if no_lags>0
rho_lags         = reshape( rho_vector(2:end) , no_lags,1); % make sure its a column vector
sumrho2          = min( sum(rho_lags.^2) , 1-rho_0^2 ); % enforce the constraint just in case
else
sumrho2 = 0;
end

%%
h_t       = h_t_with_lags(1);
h_lags    = h_t_with_lags(2:end); % should be m lags
y_t       = y_t_with_lags(1);
y_lags    = y_t_with_lags(2:end);
eta_t     = (h_t - c - phi * h_lags(1)) / sigma_eta; 
% This is mu_y as in equation (4) of Catania (2021):
epsilon_lags = (y_lags-mu) .* exp(-h_lags/2);
mu_y         = mu + exp(h_t/2) * rho_0 / ( 1 - sumrho2 ) * (eta_t - sum( rho_lags .* epsilon_lags ));
% This is sigma_y as in equation (4) of Catania (2021):
sigma_y   = exp(h_t/2) * sqrt( 1 - rho_0^2/ (1-sumrho2) );
% put this mu_y and sigma_y in a normal density:
output    = log(pdf_normal(y_t,mu_y,sigma_y));
end

