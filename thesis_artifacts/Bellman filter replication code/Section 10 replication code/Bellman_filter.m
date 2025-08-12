function [a,Info,predicted_a,predicted_Info,no_iterations,no_intervention_1,no_intervention_2] = Bellman_filter(y,psi,max_iterations)

%% Extract parameter values
mu               = psi(1);
c                = psi(2);
phi              = psi(3);
phi              = max(min(phi,0.999),-.999);
sigma_eta        = abs(psi(4));
rho_vector       = psi(5:end);
rho_vector       = reshape( rho_vector , length(rho_vector) , 1);  % make sure its a column vector
no_lags          = length(rho_vector)-1;
if no_lags>0
rho_lags         = reshape( rho_vector(2:end) , no_lags,1); % make sure its a column vector
end

%% Subtract mu from y so that from this point onward we can ignore mu
y   = y-mu;
mu  = 0;
psi = [mu;c;phi;sigma_eta;rho_vector];

%% Extract length of the data
t_final = length(y(1,:));

%% Prefill some stuff
predicted_a    = zeros(no_lags+1,t_final);
predicted_Info = zeros(no_lags+1,no_lags+1,t_final);
a              = zeros(no_lags+1,t_final);
Info           = zeros(no_lags+1,no_lags+1,t_final);
no_iterations  = zeros(1,t_final);
no_intervention_1 = 0;
no_intervention_2 = 0;

%% If the restriction is not satisfied, produce empty output
if sum(rho_vector.^2)>1
    disp('I am exiting the Bellman filter because constraint not satisfied')
else % continue with filtering

%% Get unconditional values for h
mu_h        = c / (1 - phi);

%% Initialise
for t = 1:no_lags
predicted_a(:,t) = mu_h * ones(no_lags+1,1);
a(:,t)           = mu_h * ones(no_lags+1,1);  
Info(:,:,t)      = 10 * eye(no_lags+1);   
predicted_Info(:,:,t) = 10 * eye(no_lags+1);   
end

%% Bellman filter optimsiation settings
precision      = 1/10^8;

for t = no_lags+1:t_final

    %% Collect some stuff in vectors
    y_t_with_lags             = flipud(y(1,t-no_lags:t)'); % = [ y_{t} ; y_{t-1} ; ... ; y_{t-m} ];
    y_lags                    = y_t_with_lags(2:end);
    h_lags                    = a(1:end-1,t-1);
    epsilon_lags              = y_lags .* exp(-h_lags/2);
    mu_eta                    = sigma_eta * sum( rho_lags .* epsilon_lags );
    predicted_a(:,t)          = [c+phi*a(1,t-1)+mu_eta;a(1:end-1,t-1)]; % vector size h_no_lags+1
    a_old                     = [predicted_a(1,t);a(1:end,t-1)]; % vector size no_lags+2
    negative_hessian          = - [ hessian2(a_old(1:end-1),y_lags,psi) , zeros(1+no_lags,1)  ; zeros(1,no_lags+2) ] +  [ zeros(1,no_lags+2) ; zeros(no_lags+1,1) , Info(:,:,t-1) ];      
    predicted_Info(:,:,t)     = negative_hessian(1:end-1,1:end-1) - negative_hessian(1:end-1,end) * pinv(negative_hessian(end,end)) * negative_hessian(end,1:end-1); % schur complement of size (m+1)*(m+1)
    
    %% Initialise optimalisation
    j=0; 
    delta=1;
    
    %% Start loop
    while and(lt(j,max_iterations),ge(delta,precision))
    intervention_dummy = 0;
    
    % Compute Hessian
    negative_hessian_Newton   = - [ hessian1(y_t_with_lags,a_old(1:end-1),psi) , zeros(1+no_lags,1)  ; zeros(1,no_lags+2) ]          - [ hessian2(a_old(1:end-1),y_lags,psi) , zeros(1+no_lags,1)  ; zeros(1,no_lags+2) ] +  [ zeros(1,no_lags+2) ; zeros(no_lags+1,1) , Info(:,:,t-1) ];   % hessian of objective function at the peak, size (m+2)*(m+2)
    negative_hessian_Fisher   = - [ hessian1_expected(y_t_with_lags,a_old(1:end-1),psi) , zeros(1+no_lags,1)  ; zeros(1,no_lags+2) ] - [ hessian2(a_old(1:end-1),y_lags,psi) , zeros(1+no_lags,1)  ; zeros(1,no_lags+2) ] +  [ zeros(1,no_lags+2) ; zeros(no_lags+1,1) , Info(:,:,t-1) ];   % hessian of objective function at the peak, size (m+2)*(m+2)
    Newton_weight0            = 1/2;
    negative_hessian          = Newton_weight0 * negative_hessian_Newton + ( 1 - Newton_weight0 ) * negative_hessian_Fisher;
    % Compute gradient
    gradient =  ( [score1(y_t_with_lags,a_old(1:end-1),psi);0] + [score2(a_old(1:end-1),y_lags,psi);0] - [0;Info(:,:,t-1)*(a_old(2:end,1)-a(:,t-1))] );
    
    % Test if the negative Hessian is psd (test1) and well conditioned,
    % i.e. not nearly singular (test2)
    [~,test1]=chol(negative_hessian);
    test2    =rcond(negative_hessian);
   
    if and(test1==0,test2>0.000001)
    a_new    = a_old + negative_hessian \ gradient;
    delta    = max(abs(a_new-a_old));
    a_old    = a_new;
    j        = j+1;
    else     
    % If Hessian is negative definite or positive definite but ill
    % conditioned, then we don't update (i.e., update=prediction)
    no_intervention_1  = no_intervention_1+1;
    a_new              = [predicted_a(1,t);a(1:end,t-1)];
    j                  = max_iterations; % forcing it to quit the loop over j 
    intervention_dummy = 1;
    if mod(t,400) == 0
    disp(['an intervention was made in optimisation step at t=',num2str(t),' and optimisation step=',num2str(j)])
    end
    % close loop over the test
    end
    % close the while loop
    end
    
    % Now that the optimisation has finished, count total number of iterations
    no_iterations(1,t)      = j;
    a(:,t)                  = a_new(1:end-1,1); % length no_lags+1
    Newton_weight           = 1;
    negative_hessian_Newton = - [ hessian1(y_t_with_lags,a_new(1:end-1),psi) , zeros(1+no_lags,1)  ; zeros(1,no_lags+2) ] - [ hessian2(a_new(1:end-1),y_lags,psi) , zeros(1+no_lags,1)  ; zeros(1,no_lags+2) ] +  [ zeros(1,no_lags+2) ; zeros(no_lags+1,1) , Info(:,:,t-1) ];   % hessian of objective function at the peak, size (m+2)*(m+2)
    negative_hessian_Fisher = - [ hessian1_expected(y_t_with_lags,a_new(1:end-1),psi) , zeros(1+no_lags,1)  ; zeros(1,no_lags+2) ] - [ hessian2(a_new(1:end-1),y_lags,psi) , zeros(1+no_lags,1)  ; zeros(1,no_lags+2) ] +  [ zeros(1,no_lags+2) ; zeros(no_lags+1,1) , Info(:,:,t-1) ];   % hessian of objective function at the peak, size (m+2)*(m+2)
    negative_hessian        = Newton_weight * negative_hessian_Newton + ( 1 - Newton_weight ) * negative_hessian_Fisher;
    
    if intervention_dummy==1
        % if an intervention was made, start again by setting Info to some
        % multiple of identity matrix
    Info(:,:,t) = 10 * eye(no_lags+1);
        % if no invernetion was made, try to compute the negative hessian
        % by Schur complement
    else
        Info(:,:,t) = negative_hessian(1:end-1,1:end-1) - negative_hessian(1:end-1,end) * pinv(negative_hessian(end,end)) * negative_hessian(end,1:end-1); % schur complement of size (m+1)*(m+1)
        % test if information is positive definite and well conditioned
        [~,test1]=chol(Info(:,:,t));
        test2 = rcond(Info(:,:,t));
    if and(test1==0,test2>0.000001)
    % do nothing
    else
    % if negative definite or ill-conditioned, make another intervention    
    Info(:,:,t)        = 10 * eye(no_lags+1);
    no_intervention_2  = no_intervention_2+1;
    disp(['intervention made in updating step at t=',num2str(t)])
    % close test
    end     
    % close intervention
    end
% Close loop over time
end
% Close if function for constraint
end
% Close entire function
end

