function [output] = score1(y_t_with_lags,h_t_with_lags,psi)

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
y_t       = y_t_with_lags(1);
y_lags    = y_t_with_lags(2:end);
eta_t     = (h_t - c - phi * h_lags(1)) / sigma_eta;
epsilon_lags = (y_lags-mu) .* exp(-h_lags/2);
% This is mu_y as in equation (4) of Catania (2021):
mu_y      = mu + exp(h_t/2) * rho_0 / ( 1 - sum_squared_rho_lags ) * (eta_t - sum( rho_lags .* epsilon_lags ));
% This is sigma_y as in equation (4) of Catania (2021):
sigma_y   = exp(h_t/2) * sqrt( 1 - rho_0^2/ (1-sum_squared_rho_lags) );

%% Now compute the score using the chain rule (via both mu_y and sigma_y) 
u_t                 = ( y_t - mu_y ) / sigma_y ;
score_wrt_mu        = u_t / sigma_y ;
score_wrt_sigma     = ( u_t^2 - 1 ) / sigma_y;
gradient_of_mu_y    = [(mu_y - mu)/2;zeros(no_lags,1)] + rho_0 / (1-sum_squared_rho_lags) * exp(h_t/2)  * ( [ 1/sigma_eta ; -phi/sigma_eta ; zeros(no_lags-1,1) ] + [ 0 ; 1/2 * rho_lags .* epsilon_lags ] );
gradient_of_sigma_y = 1/2 * sigma_y; 
output              = score_wrt_mu * gradient_of_mu_y + [score_wrt_sigma * gradient_of_sigma_y;zeros(no_lags,1) ];

end

