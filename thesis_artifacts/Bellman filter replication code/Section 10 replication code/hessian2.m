function [output] = hessian2(h_t_with_lags,y_lags,psi)

%% Extract parameter values
mu               = psi(1);
c                = psi(2);
phi              = psi(3);
phi              = max(min(phi,0.995),-.995);
sigma_eta        = psi(4);
rho_vector       = psi(5:end);
rho_vector       = reshape( rho_vector , length(rho_vector) , 1); % make sure its a column vectors
rho_0            = rho_vector(1);
no_lags          = length(rho_vector)-1;
if no_lags>0
rho_lags         = reshape( rho_vector(2:end) , no_lags,1); % make sure its a column vector
sum_squared_rho_lags = min( sum(rho_lags.^2) , 1-rho_0^2 ); % enforce the constraint just in case
else
sum_squared_rho_lags = 0;
end

%%
h_t       = h_t_with_lags(1);
h_lags    = h_t_with_lags(2:end); % there should be m lags
%eta_t     = (h_t - c - phi * h_lags(1)) / sigma_eta;
epsilon_lags = (y_lags-mu) .* exp(-h_lags/2);
% This is mu_h as in equation (4) of Catania (2021):
mu_h      = c + phi * h_lags(1) + sigma_eta * sum( rho_lags .* epsilon_lags );
% This is sigma_h as in equation (4) of Catania (2021):
sigma_h   = sigma_eta * sqrt( 1 - sum_squared_rho_lags );
% Timplies state-transition shock

%% Now compute the score using the chain rule (via both mu_y and sigma_y) 
w_t              = ( h_t - mu_h ) / sigma_h;
score_wrt_mu_h   = w_t / sigma_h;
hessian_wrt_mu_h = - 1 / sigma_h^2;
term1            = [-1;phi;zeros(no_lags-1,1)];
term2            = -1/2 * sigma_eta * [0; rho_lags .* epsilon_lags ];
gradient_of_mu_h = term1 + term2 ;
output           = hessian_wrt_mu_h * (gradient_of_mu_h * gradient_of_mu_h') + (1/4) * sigma_eta * score_wrt_mu_h * diag([0;rho_lags.*epsilon_lags]);

end

