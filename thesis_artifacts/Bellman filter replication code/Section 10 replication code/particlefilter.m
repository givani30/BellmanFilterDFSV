function [output_LLK, updatedh, predictedh, updatedmedian, predictedmedian, updatedmean, predictedmean] = particlefilter(parameters, y, state_errors, unif_draws, init_draw)

N           = size(state_errors,1);
T           = size(y,2);

mu          = parameters(1); 
omega       = parameters(2);
phi         = parameters(3);
sigma_eta   = parameters(4);
rho_vec     = parameters(5:end);
y           = y - mu; % 
rho0        = rho_vec(1);

if sum(rho_vec.^2) >= 1 || phi >= 1 || phi <= 0 || sigma_eta <= 0
     output_LLK = inf;
else
m           = length(rho_vec) - 1;
iRow        = m+1;
if m == 0; iRow = 2; end
if m > 0;  sumrho2 = sum(rho_vec(2:end).^2); end

mH          = zeros(N,iRow);
LLK         = zeros(T,1);
weights     = zeros(N,1);

% Calculate unconditional distribution of h
term        = 0;
for i = 1:m
    term    = term + phi^i * sum(rho_vec(1:end-i) .* rho_vec(i+1:end));
end

sigma2_h    = sigma_eta^2 / (1 - phi^2) * (1 + 2 * term);
sigma_h     = sqrt(sigma2_h);
mu_h        = omega / (1 - phi); 

predictedh      = zeros(T,1);
updatedh        = zeros(T,1);
updatedmedian   = zeros(T,1);
predictedmedian   = zeros(T,1);
updatedmean   = zeros(T,1);
predictedmean   = zeros(T,1);

% Compute intial draws t0
mH(:,1)     = mu_h + sigma_h * init_draw;
mH(:,2)     = mH(:,1);

% Define terms needed in multiple computations
if m > 0
    scale_y = sqrt(1 - rho0^2 / (1 - sumrho2));
    scale_h = sqrt(1 - sumrho2);
else
    scale_y = sqrt(1 - rho0^2);
    scale_h = 1;
end

for t = 1:T    
    yh_term = zeros(N,1);
    if m > 0
        for j = 1:m
            if t >= j
                if t-j < 1
                    yh_term = yh_term + rho_vec(j+1) * y(t) .* exp(-0.5 * mH(:,j+1));
                else
                    yh_term = yh_term + rho_vec(j+1) * y(t-j) .* exp(-0.5 * mH(:,j+1));
                end

            end
        end
    end
    % Predict
    mH(:,1)     = omega + phi * mH(:,2) + sigma_eta * yh_term(:) + sigma_eta * scale_h .* state_errors(:,t);
    % Compute weights
    mu_y        = exp(mH(:,1) ./ 2) .* rho0 / scale_h .* state_errors(:,t);
    sigma_y     = exp(mH(:,1) ./ 2) .* scale_y;
    weights(:)  = pdf_normal(y(t), mu_y, sigma_y);

    % Predict state by taking expected value
    predictedh(t)       = sum(mH(:,1)) / N;
    predictedmedian(t)  = median(link(mH(:,1)));
    predictedmean(t)    = sum(link(mH(:,1))) / N;

    % Log likelihood contribution
    sumweights  = sum(weights);
    LLK(t)      = log(sumweights/N);

    % Normalised weights
    weights_nor = weights / sumweights;

    % Resample
    mH(:,1)          = resample(mH(:,1), weights_nor, unif_draws(t));

    if m == 0
        mH(:,2)     = mH(:,1); 
    else
        mH(:,2:end) = mH(:,1:end-1);
    end

    updatedh(t)         = sum(mH(:,1)) / N;
    updatedmedian(t)    = median(link(mH(:,1)));
    updatedmean(t)      = sum(link(mH(:,1))) / N;
end

output_LLK = - sum(LLK);
end
end