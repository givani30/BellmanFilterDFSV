function [a,Info,predicted_a,predicted_Info,no_iterations,delta_stored] = Bellman_filter1(data,parameter_vector,max_iterations)

%% Get the size of the data
m       = size(data,1); % cross section
t_final = size(data,2); % time dimension

%% Get the parameters from a row vector
gamma1 = parameter_vector(1);
gamma2 = parameter_vector(2);
gamma3 = parameter_vector(3);
tau    = max(abs(parameter_vector(4)),0.00001); %force to be positive
lambda = max(abs(parameter_vector(5)),0.00001); %force to be positive
cons   = parameter_vector(6);  

%% Create a tri-diagonal transfer martrix T
T = gamma2*diag(ones(1,m-1),1) + gamma1*diag(ones(1,m),0) + gamma3*diag(ones(1,m-1),-1);

%% Create more relevant matrices for easy reference
identity    = eye(m);

%% Create Q matrix
[i,j] = meshgrid(1:m);
Q = tau^2 * (1+sqrt(3)*abs(i-j)/lambda).*exp(-sqrt(3)*abs(i-j)/lambda);
Q = (Q>1e-9).*Q;
Qinv = identity/Q; % this is equivalent to Qinv = inv(Q) but quicker;
%Qinv = inv(Q);
clear i j

%% Construct vector c
c = cons * ones(m,1);

%% Prefill some stuff to be used below
a               = zeros(m,t_final); % this will store our filtered states
predicted_a     = zeros(m,t_final); % this will store our predicted states
Info            = zeros(m,m,t_final); % this will store filtered Info matrix
predicted_Info  = zeros(m,m,t_final); % this will store predicted Info matrix
no_iterations   = zeros(1,t_final); % this will store how many iterations were needed at each time step for convergence
delta_stored    = zeros(max_iterations,t_final); % this will store, at each time step, the maximum difference between old and new states
rho_stored      = zeros(max_iterations,t_final); % this will store, at each time step, all the rho values used by the BFGS algorithm

%% Initialise at time step one
predicted_a(:,1)       = cons * ones(m,1); % unconditional mean
predicted_Info(:,:,1)  = 1/100*eye(m); % close to `uninformative', i.e. close to diffuse

%% Recursive optimisation procedure
precision    = 1/10^5; % precision for each optimisation at each time step  
intervention = 0; % initially set this to zero. it will be set to one if an intervention occurs.
step_grid    = (0:1/100:1); % this grid is used for a line search in the optimisation
n            = length(step_grid); % number of gridpoints in the line search

%% Then run the Bellman filter
for t=1:t_final

  %% Optimisation for each time step
    j=1; % j counts the number of optimisation steps
    delta=1; % delta keeps track of the difference between old and new estimates

    while and(le(j,max_iterations),ge(delta,precision))
        % for the first optimisation step
        if j==1
            a_old     = predicted_a(:,t); 
            % test if the updated info is well defined and p.d.
            [~,test1] = chol( predicted_Info(:,:,t)+info(data(:,t),a_old) );
            test2     = rcond( predicted_Info(:,:,t)+info(data(:,t),a_old) );
            % if it is well defined, compute its inverse
            if and(test1==0,and(~isnan(test2),test2>eps))
            InverseNegHessian = inv(predicted_Info(:,:,t)+info(data(:,t),a_old)); 
            else
            % if it is not well defined, set intervention dummy to one and
            % use identity matrix 
            intervention = 1;
            InverseNegHessian = identity; 
            end
        end
    % compute the score and the gradient of the objective function 
    score_old         = score(data(:,t),a_old);
    gradient_old      = score_old - predicted_Info(:,:,t) * (a_old - predicted_a(:,t));
    % compute the step direction based on the inverse negative hessian and
    % gradient:
    step_direction    = InverseNegHessian * gradient_old;
    % consider several a_new for all steps sizes on the grid
    a_new             = a_old + step_grid .* step_direction; % here we are adding things of different sizes, but it works fine
    % compute the value of the objective function for all stepsizes
    values            = logpdf(data(:,t),a_new) - 1/2 * diag( transpose(a_new -repmat(predicted_a(:,t),1,n)) * predicted_Info(:,:,t) * (a_new -  repmat( predicted_a(:,t),1,n)))';
    % select the the id of the stepsize that gives the best value for the objective
    [~,id] = max(values);
    % use the best stepsize in doing the actual update
    a_new             = a_old + step_grid(id) * step_direction;
    a_dif             = a_new - a_old;
    score_new         = score(data(:,t),a_new);
    % Compute the gradient at the new point that was just found:
    gradient_new      = score_new - predicted_Info(:,:,t) * (a_new - predicted_a(:,t));
    % Compute the rho value that is used in the BFGS algorithm:
    gradient_dif      = gradient_new - gradient_old;
    rho               = 1 / min(gradient_dif' * a_dif,-1/10^10); % this number should be negative
    rho_stored(j,t)   = rho;
    % BFGS update of the the InverseNegativeHessian
    InverseNegHessian = (identity - rho * a_dif * gradient_dif') * InverseNegHessian * (identity - rho * gradient_dif * a_dif') - rho * (a_dif * a_dif');
    delta             = max(abs(a_new-a_old));
    delta_stored(j,t) = delta;
    a_old             = a_new;
    j                 = j+1;
    end
    % Test if we actually converged
    if or(delta>precision,isnan(delta))
    %    disp(['failed to converge at time=',num2str(t)]) 
    intervention=1;
    end
    
    %% Store the number of iterations that were needed
    no_iterations(1,t) = j-1;   
    
    %% Do the updating step of the filter
    a(:,t)         = a_new;
    Info(:,:,t)    = predicted_Info(:,:,t) + info(data(:,t),a_new);

    %% Predict the level
    predicted_a(:,t+1)       = (identity-T) * c + T * a(:,t);

    %% Predict information
    [~,test1]         = chol(Info(:,:,t) + T' * Qinv * T);
    test2             = rcond(Info(:,:,t) + T' * Qinv * T);
    % If the Hessian is fine, do the Newton step
    if and(test1==0,and(~isnan(test2),test2>eps))
    predicted_Info(:,:,t+1) = Qinv - Qinv * T * (( Info(:,:,t) + T' * Qinv * T ) \ (T' * Qinv));
    else
    intervention = 1;
    predicted_Info(:,:,t+1) = predicted_Info(:,:,1); % restart with unconditional distribution
    end
% Close the loop over time
end

% Delete last entry in prediction
predicted_a    = predicted_a(:,1:end-1);
predicted_Info = predicted_Info(:,:,1:end-1);

% If any intervention was made, give a warning
if and(intervention==1,mod(t,10)==0)
    disp('BFGS: some interventions were made')
end

% Close function
end

