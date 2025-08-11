function [NegativeLogL,LL1,LL2,LL3]=NegativeLogLikelihood1_with_c_set_to_zero(parameter_vector,data,max_iterations)

%% Get the size of the data
m       = size(data,1);
t_final = size(data,2);

%% Run the Bellman filter
[filtered_a,filtered_Info,predicted_a,predicted_Info] = Bellman_filter1_with_c_set_to_zero(data,parameter_vector,max_iterations);

%% Prefill some stuff
LL = zeros(1,t_final);
LL1 = zeros(1,t_final);
LL2 = zeros(1,t_final);
LL3 = zeros(1,t_final);
a   = zeros(m,t_final);

%% Calculate the log likelihood of the data
for t=1:t_final
    a(:,t)      =  filtered_a(:,t);
    LL1(t)      =  logpdf(data(:,t),a(:,t)) ;
    LL2(t)      =  1/2 * sum( log(eig( predicted_Info(:,:,t) )))   - 1/2*(a(:,t)-predicted_a(:,t))'* predicted_Info(:,:,t)  * (a(:,t)-predicted_a(:,t));
    LL3(t)      =  1/2 * sum( log(eig( filtered_Info(:,:,t)  )))   - 1/2*(a(:,t)-filtered_a(:,t) )'* filtered_Info(:,:,t)   * (a(:,t)-filtered_a(:,t) );
    LL(t)       =  LL1(t) + LL2(t) - LL3(t);
end

%% Sum up the different parts
NegativeLogL  = - sum(LL(1:end));
%disp([parameter_vector,NegativeLogL/10^3])

% close the function
end