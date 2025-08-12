function [output] = hessian1(y_t_with_lags,h_t_with_lags,psi)

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

% Standardised residual
u_t               = ( y_t - mu_y ) / sigma_y ;

% First and second derivatives of observation density with respect to mu_y
score_wrt_mu_y      = u_t / sigma_y ;
hessian_wrt_mu_y    = -1 / sigma_y^2 ;

% First and second derivatives of observations density with respect to sigma_y
score_wrt_sigma_y   = ( u_t^2 - 1 ) / sigma_y;
hessian_wrt_sigma_y = ( 1 - 3 * u_t^2 ) / sigma_y^2;

% Cross derivative with respect mu_y and sigma_y
cross_hessian = - 2 * u_t / sigma_y^2;

% First derivative of mu_y with respect to h
gradient_of_mu_y = [(mu_y - mu)/2;zeros(no_lags,1)] + rho_0 / (1-sum_squared_rho_lags) * exp(h_t/2)  * ( [ 1/sigma_eta ; -phi/sigma_eta ; zeros(no_lags-1,1) ] + [ 0 ; 1/2 * rho_lags .* epsilon_lags ] );

% Second derivative of mu_y with respect to h
term0            = (mu_y - mu)/4 * diag([1;zeros(no_lags,1)]);
term1            = rho_0 / (1-sum_squared_rho_lags) * exp(h_t/2) *          diag([1/sigma_eta;zeros(no_lags,1)]);
term2            = rho_0 / (1-sum_squared_rho_lags) * exp(h_t/2) * (-1/4) * diag([0;rho_lags.*epsilon_lags]); 
term3            = 1/2 * [0, gradient_of_mu_y(2:end)' ; gradient_of_mu_y(2:end) , zeros(no_lags,no_lags)];
hessian_of_mu_y  = term0 + term1 + term2 + term3;

% First derivative of sigma_y with respect to h
gradient_of_sigma_y = [sigma_y/2;zeros(no_lags,1)];

% Second derivative of sigma_y with respect to h
hessian_of_sigma_y = zeros(no_lags+1,no_lags+1);
hessian_of_sigma_y(1,1) = sigma_y/4;

% Compute output via the chain rule
term1 = [gradient_of_mu_y , gradient_of_sigma_y] * [ hessian_wrt_mu_y , cross_hessian ; cross_hessian , hessian_wrt_sigma_y ] * [gradient_of_mu_y' ; gradient_of_sigma_y'] ;
term2 = score_wrt_sigma_y * hessian_of_sigma_y;
term3 = score_wrt_mu_y    * hessian_of_mu_y;
output = term1 + term2 + term3;

end

