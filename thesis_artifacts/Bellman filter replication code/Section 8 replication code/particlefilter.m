function [output_LLK, updateda_mean , updateda_median, predictedlinka_mean] = particlefilter(parameters, y, state_errors, unif_draws, init_draw)


N           = size(state_errors,2);
T           = size(state_errors,3);
state_errors = reshape(state_errors, N, T);
init_draw    = reshape(init_draw, N, 1);
% For copula
d = size(y,3);
if d == 2
    y = reshape(y,T,2)';
end

c           = parameters(1);
phi         = parameters(2);
sigma_eta   = parameters(3);

if length(parameters) > 3
    extra_par   = parameters(4:end);
else
    extra_par   = nan;
end

% X          = zeros(N,1);
LLK          = zeros(T,1);
% weights     = zeros(N,1);

% Calculate unconditional distribution of h
sigma2_a    = sigma_eta^2 / (1 - phi^2);
sigma_a     = sqrt(sigma2_a);
mu_a        = c / (1 - phi);

%predictedh_mean = zeros(T,1);
predicteda_median     = zeros(T,1);
predictedlinka_mean   = zeros(T,1);
updateda_mean         = zeros(T,1);
updateda_median       = zeros(T,1);

% Compute intial draws t0
X = mu_a + sigma_a * init_draw;

for t = 1:T
    X           = c + phi * X + sigma_eta * state_errors(:,t);
    weights     = pdf(y(:,t), X, extra_par);
    
    %predictedh_mean(t)       = sum(X) / N;
    predicteda_median(t)      = median(X);

    predictedlinka_mean(t)    = sum(link(X)) / N;
    %predictedlink_median(t)  = median(link(X)) ;
    
    % Log likelihood contribution
    sumweights  = sum(weights);
    LLK(t)      = log(sumweights/N);
    
    % Normalised weights
    weights_nor = weights / sumweights;
    
    % Resample
    X          = resample(X, weights_nor, unif_draws(1,t));

    updateda_mean(t)     = sum(X) / N;
    updateda_median(t)   = median(X);
    
    %updatedlink_mean(t)    = sum(link(X)) / N;
    %updatedlink_median(t)  = median(link(X)) ;
end
    

output_LLK = - sum(LLK);
end