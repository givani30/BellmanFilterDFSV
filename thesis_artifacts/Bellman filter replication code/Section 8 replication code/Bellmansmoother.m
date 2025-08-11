function [smootheda, smoothedP] = Bellmansmoother(y, parameters, max_iterations,modeltype)

% Extract length of the data
t_final = size(y,2);

% Extract the stuff we need from the input arguments
%c               = parameters(1);
phi              = min(max(parameters(2),-0.995),0.995);
%sigma_eta       = abs(parameters(3));
%Q               = sigma_eta ^ 2;
%Qinv            = 1 / Q;
%if length(parameters) > 3
%    shape_parameter   = parameters(4:end);
%else
%    shape_parameter   = nan;
%end

% Run the filter
[updateda, updatedI, predicteda, predictedI]    = Bellmanfilter(y, parameters, max_iterations,modeltype);

% Invert some output
updatedP        = 1./updatedI;
predictedP      = 1./predictedI;

% Initiliase the smoother
smootheda       =   zeros(1,t_final);
smoothedP       =   zeros(1,t_final);

% Backward loop for smoother
for tau=1:t_final
    t = t_final - (tau-1);
    if t==t_final
        smootheda(t)   = updateda(t);
        smoothedP(t)   = updatedP(t);
    else
        smootheda(t) = updateda(t) + updatedP(t) * phi' * predictedI(t+1) * (smootheda(t+1) - predicteda(t+1) );  
        smoothedP(t) = updatedP(t) - updatedP(t) * phi' * predictedI(t+1) * (predictedP(t+1) - smoothedP(t+1)) * predictedI(t+1) * phi * updatedP(t);  
    end
end
% close the function
end

