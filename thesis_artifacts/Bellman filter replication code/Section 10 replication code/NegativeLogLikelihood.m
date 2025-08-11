%%Calculated the likelihood for the model
function [output,LL]=NegativeLogLikelihood(psi,y,max_iterations)

%disp(psi)

%% Get the parameters from a column vector psi
mu              = psi(1);
c               = psi(2);
phi             = psi(3);
phi             = max(min(phi,0.999),-.999);
sigma_eta       = abs(psi(4));
rho_vector      = psi(5:end);
rho_vector      = reshape( rho_vector , length(rho_vector) , 1);  % reshape to make sure its a column vector
no_lags         = length(rho_vector)-1;

%% Check that the constraint is satisfied
if sum(rho_vector.^2)>0.99999
    output = inf;
else

%% Derive the size of the data
t_final    = length(y(1,:));

%% Run the Bellman filter
[a,Info,predicted_a,predicted_Info] = Bellman_filter(y(1,:),psi,max_iterations);

%% Subtract mu from y so that from this point onward we can ignore mu
y   = y-mu;
mu  = 0;
psi = [mu;c;phi;sigma_eta;rho_vector];

%% Compute
LL    = zeros(1,t_final);
term1 = zeros(1,t_final);
term2 = zeros(1,t_final);
term3 = zeros(1,t_final);

for t=no_lags+1:t_final
    y_t_with_lags    =  flipud(y(1,t-no_lags:t)');
    term1(t)         =  logpdf(y_t_with_lags,a(:,t),psi);
    term2(t)         =  1/2 * log( max( det( predicted_Info(:,:,t)) ,1/1000))   - 1/2*(a(:,t)-predicted_a(:,t))'* predicted_Info(:,:,t)  * (a(:,t)-predicted_a(:,t));
    term3(t)         =  1/2 * log( max( det( Info(:,:,t) ) ,1/1000))            - 1/2*(a(:,t)-a(:,t) )'         * Info(:,:,t)            * (a(:,t)-a(:,t) );
    LL(t)            =  term1(t) + term2(t) - term3(t);
        if LL(t)==isnan(NaN)
        disp('error')
        end
end

% Sum to get the output
output  = - sum( LL ) ;

% Close if statement with contstraint
end
% Close the function
end