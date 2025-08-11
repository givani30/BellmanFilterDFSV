function [a,Info,predicted_a,predicted_Info,no_iterations,delta_stored,smoothed_a] = Bellman_filter2_with_c_set_to_zero(data,parameter_vector,max_iterations)

%% Get the size of the data
m       = size(data,1);
t_final = size(data,2);

%% Get the parameters from a row vector
gamma1 = parameter_vector(1);
gamma2 = parameter_vector(2);
gamma3 = parameter_vector(3);
tau    = max(abs(parameter_vector(4)),0.00001);%force to be positive
lambda = max(abs(parameter_vector(5)),0.00001);%force to be positive
cons   = 0;
sigma  = max(abs(parameter_vector(6)),0.0001);%force to be positive

%% Create a tri-diagonal transfer martrix T
T = gamma2*diag(ones(1,m-1),1) + gamma1*diag(ones(1,m),0) + gamma3*diag(ones(1,m-1),-1);

%% Create Q matrix
[i,j] = meshgrid(1:m);
Q = tau^2 * (1+sqrt(3)*abs(i-j)/lambda).*exp(-sqrt(3)*abs(i-j)/lambda);
Q = (Q>1e-9).*Q;
clear i j

%% Construct vector c
c = cons * ones(m,1);

%% Create R
R = sigma^2 * eye(m);

%% Create more relevant matrices
zero_vector = zeros(m,1);
zero_matrix = zeros(m,m);
identity    = eye(m);
big_identity= eye(2*m);
TT          = [zero_matrix,identity;zero_matrix,T];
QQ          = [R,zero_matrix;zero_matrix,Q];
Qinv        = eye(2*m) / QQ; % equivalent to Qinv       = inv(Q);

%% Prefill filtered quantities
a               = zeros(2*m,t_final);
predicted_a     = zeros(2*m,t_final);
Info            = zeros(2*m,2*m,t_final);
predicted_Info  = zeros(2*m,2*m,t_final);
no_iterations   = zeros(1,t_final);
delta_stored    = zeros(max_iterations,t_final);

%% Initialise
%predicted_a(:,1)      = (big_identity - T) \ [zero_vector;(identity-M)*c]; %this should be cons
predicted_a(:,1)       = cons * ones(2*m,1); 
%predicted_Info(:,:,1) = inv(reshape( ( eye(4*N*N)-kron(T,T) ) \ reshape(Q,4*N*N,1),2*N,2*N));
predicted_Info(:,:,1)  = 1/100*eye(2*m);

%% Recursive optimisation procedure
precision    = 1/10^5; % precision for each optimisation at each time step  
intervention1= 0; % initially set this to zero. it will be set to one if an intervention occurs.
intervention2= 0; % initially set this to zero. it will be set to one if an intervention occurs.
step_grid    = (0:1/100:1); % this grid is used for a line search in the optimisation
n            = length(step_grid); % number of gridpoints in the line search

%% Then run the Bellman filter
for t=1:t_final

  %% Optimisation for each time step
    j=1; % j counts the number of optimisation steps
    delta=1; % delta keeps track of the difference between old and new estimates

    while and(le(j,max_iterations),ge(delta,precision))
    
     %% For first iteration:
        if j==1
        a_old             = predicted_a(:,t); 
        information       = [info(data(:,t),a_old(1:m)),zero_matrix;zero_matrix,zero_matrix];
             % test if the updated info is well defined and p.d.
        [~,test1]         = chol( predicted_Info(:,:,t)+information );
        try
            test2             = rcond( predicted_Info(:,:,t)+information );
        catch
            test2 = nan;
        end
        % if it is well defined, compute its inverse
         if and(test1==0,and(~isnan(test2),test2>eps))
            InverseNegHessian = inv(predicted_Info(:,:,t)+information); 
         else
            % if it is not well defined, set intervention dummy to one and
            % use identity matrix 
            intervention1 = 1;
            InverseNegHessian = eye(2*m); 
         end
        end

     %% Determine quantities to iterate 
    score_old         = score(data(:,t),a_old(1:m));
    gradient_old      = [score_old;zero_vector] - predicted_Info(:,:,t) * (a_old - predicted_a(:,t));
    step_direction    = InverseNegHessian * gradient_old;
    
    % consider several a_new for all steps sizes on the grid
    a_new             = a_old + step_grid .* step_direction;

    % compute the value of the objective function for all stepsizes
    values            = logpdf(data(:,t),a_new(1:m,:)) - 1/2 * diag( transpose(a_new -repmat(predicted_a(:,t),1,n)) * predicted_Info(:,:,t) * (a_new -  repmat( predicted_a(:,t),1,n)))';
    
    % select the best stepsize
    [~,id] = max(values);

    % Use the best stepsize in doing the actual update
    a_new             = a_old + step_grid(id) * step_direction;
    a_dif             = a_new - a_old;
    score_new         = score(data(:,t),a_new(1:m));
    gradient_new      = [score_new;zero_vector] - predicted_Info(:,:,t) * (a_new - predicted_a(:,t));
    gradient_dif      = gradient_new - gradient_old;
    rho               = 1 / min(gradient_dif' * a_dif,-1/10^10); % this number should be negative
    
    %% BFGS update
    InverseNegHessian = (big_identity - rho * a_dif * gradient_dif') * InverseNegHessian * (big_identity - rho * gradient_dif * a_dif') - rho * (a_dif * a_dif');
    delta             = max(abs(a_new-a_old));
    delta_stored(j,t) = delta;
    a_old             = a_new;
    j                 = j+1;
    end

    % Test if we actually converged
    if delta>precision
        intervention2=1;
        disp(['failed to converge at time=',num2str(t)]) 
    end

    %% Store number of iterations needed
    no_iterations(1,t) = j-1;
       
    %% Update
    a(:,t)         = a_new;
    Info(:,:,t)    = predicted_Info(:,:,t) + [info(data(:,t),a(1:m,t)),zero_matrix;zero_matrix,zero_matrix];
    
    %% Predict level
    predicted_a(:,t+1)       = [zero_vector;(identity-T)*c] + TT * a(:,t);

    %% Predict information
    [~,test1] = chol(Info(:,:,t) + TT' * Qinv * TT);
    try
        test2 = rcond(Info(:,:,t) + TT' * Qinv * TT);
    catch
        test2 = nan;
    end
    % If the Hessian is fine, do the Newton prediction step
    if and(test1==0,and(~isnan(test2),test2>eps))
    predicted_Info(:,:,t+1)  = Qinv - (Qinv * TT) * (( Info(:,:,t) + TT' * Qinv * TT ) \ (TT' * Qinv));
    else
    intervention2 = 1;
    predicted_Info(:,:,t+1) = predicted_Info(:,:,1); % restart with unconditional distribution
    %disp(['Hierarchical BFGS version: there were errors in the prediction step with condition number=',num2str(test2)])
    end
 
% Close the loop over time
end

%% Do one-step smoothing (use RTS smoother but for one step only)
smoothed_a = zeros(2*m,t_final);
for t=1:t_final
    if t<t_final
    smoothed_a(:,t)=a(:,t) + Info(:,:,t) \ ( TT' * predicted_Info(:,:,t+1) * ( a(:,t+1) - predicted_a(:,t+1) ));
    else
    smoothed_a(:,t)=a(:,t);
    end
end

% take only the lower part
smoothed_a = smoothed_a(m+1:2*m,:);

%% Delete last entrance in prediction
predicted_a    = predicted_a(:,1:t_final);
predicted_Info = predicted_Info(:,:,1:t_final);

%% Display a warning in case interventions were made
if intervention1==1
    disp('Hierarchical BFGS version: there were errors in the intitialisation of the optimisation')
end
if intervention2==1
    disp('Hierarchical BFGS version: there were errors in the prediction step')
end

%% Close function
end

