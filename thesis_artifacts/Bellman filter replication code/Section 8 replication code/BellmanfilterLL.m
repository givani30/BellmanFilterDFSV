function [output_LL,LL] = BellmanfilterLL(parameters, y, max_iter,modeltype)

% Extract length of the data
t_final = size(y,2);
t0 = 1; % We use unconditional distribution

%% Extract the stuff we need from the input arguments
%c              = parameters(1);
phi             = parameters(2);
sigma_eta       = parameters(3);
%Q               = sigma_eta^2;
%Qinv            = 1 / Q;

if length(parameters) > 3
    shape_parameter   = parameters(4:end);
else
    shape_parameter   = nan;
end

%% Impose constraint that phi needs to be smaller than one in magintude
if ge(phi,.995) || le(phi,-.995) || le(sigma_eta,0.001)
    output_LL = inf;
else

%% Run the Bellman filter
[updateda, updatedI, predicteda, predictedI]    = Bellmanfilter(y, parameters, max_iter,modeltype);

%% Compute pseudo log likelihood
LL = zeros(t_final-t0,1);
for t = t0:t_final
    term1       = logpdf(y(:,t), updateda(t), shape_parameter);
    term2       = 0.5 * log( det( updatedI(t) \ predictedI(t) ));
    term3       = 0.5 * ( updateda(t) - predicteda(t) )' * predictedI(t) *  ( updateda(t) - predicteda(t) );   
    LL(t-t0+1)  = term1 + term2 - term3;
end
output_LL = - sum(LL);

%% close if statement
end
%% close function
end

