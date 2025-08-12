function [updateda, updatedI, predicteda, predictedI] = Bellmanfilter(y, parameters, max_iterations,modeltype)

%% Extract length of the data
t_final = size(y,2);

%% Extract the stuff we need from the input arguments
c               = parameters(1);
phi             = min(max(parameters(2),-0.995),0.995);
sigma_eta       = max(parameters(3),0.001);
Q               = sigma_eta ^ 2;
Qinv            = 1 / Q;
if length(parameters) > 3
    shape_parameter   = parameters(4:end);
else
    shape_parameter   = nan;
end

%% Set the weight to be used for the Fisher updating step
if isequal(modeltype,'SCg')
    Fisher_optimisation_weight = 1/2;
    Fisher_updating_weight     = 1/2;
elseif isequal(modeltype,'SCt')
    nu = min(max(shape_parameter(2),4),40);
    Fisher_optimisation_weight = 1/2 * (nu+4)/(nu+3);
    Fisher_updating_weight     = 1/2 * (nu+4)/(nu+3);
elseif isequal(modeltype,'LLt')
    nu = min(max(shape_parameter(2),4),40);
    Fisher_optimisation_weight = 0;
    Fisher_updating_weight = (1+nu/3) / (1+3*nu);
else
    Fisher_optimisation_weight= 1/2;
    Fisher_updating_weight    = 0;
end
  
%% Unconditional initialisation
a0 = c / (1-phi);
I0 = (1 - phi^2) / Q;

predictedI      =   zeros(1,t_final);
predicteda      =   zeros(1,t_final);
updatedI        =   zeros(1,t_final);
updateda        =   zeros(1,t_final);

%% Set precision
precision = 1/10^5;

%% Bellman filter loop
for t =1:t_final
    %disp(t)
    %Prediction step
    if t == 1
        predicteda(t)   = c + phi * a0;
        predictedI(t)   = Qinv - Qinv * phi / (I0 + phi' * Qinv * phi) * phi * Qinv;
    else
        predicteda(t)   = c + phi * updateda(t-1);
        predictedI(t)   = Qinv - Qinv * phi / (updatedI(t-1) + phi' * Qinv * phi) * phi * Qinv;
    end
    
    %Start
    a       = predicteda(t);
    i       = 1;
    delta   = 1;
    %Optimise Newton
    while(le(i,max_iterations) && ge(delta,precision))
        real_info   = realinfo(y(:,t),a,shape_parameter);
        exp_info    = expinfo(y(:,t),a,shape_parameter);
        info        = (1-Fisher_optimisation_weight) * real_info + Fisher_optimisation_weight * exp_info;
        score_value = score(y(:,t),a,shape_parameter);
        a_new       = a + (predictedI(t) + info) \ (score_value - predictedI(t) * (a - predicteda(t)));
        delta       = abs(a_new-a);
        a           = a_new;
        i           = i+1;
    end
    updateda(t)     = a;
    
    %Update Info: weighted version of Newton and Fisher
    real_info      = realinfo(y(:,t),a,shape_parameter);
    exp_info       = expinfo(y(:,t),a,shape_parameter);
    update_info    = (1-Fisher_updating_weight) * real_info + Fisher_updating_weight * exp_info;
    updatedI(t)    = predictedI(t) + update_info;

end

%% close the function
end
