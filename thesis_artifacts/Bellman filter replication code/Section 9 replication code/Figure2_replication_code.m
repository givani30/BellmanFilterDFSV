%% Clear and close
clear
close all

%% Automatically get the folder name where this file is stored 
file_name_full   = matlab.desktop.editor.getActiveFilename;
separator_id     = strfind(file_name_full,'/');
file_name_short  = file_name_full(max(separator_id)+1:end);
folder_name      = erase(file_name_full,file_name_short);

%% If the above doesn't work, you can also "hard code" the folder name
%folder_name = '/Users/Rutger-Jan/Dropbox (Erasmus Universiteit Rotterdam)/'

%% Tell Matlab to load the current folder
cd(folder_name);

%% Size of the data (change this depending on the simulation setup)
m       = 90; % the article considers m=10,30,60,90,120,150
t_final = 80;

%% Size of the simulation study
number_of_samples       = 100;
no_particles            = 10^4; % the article considers 10^4, 10^5 and 10^6
rng(123)

%% True parameters = [ gamma1, gamma2, gamma3, tau, lambda, sigma_xi, c]
true_parameters = [  0.4   0    0.4    0.8    5   0];

% Construct T
gamma1 =  true_parameters(1);
gamma2 =  true_parameters(2);
gamma3 =  true_parameters(3);
T      =  gamma2*diag(ones(1,m-1),1) + gamma1*diag(ones(1,m),0) + gamma3*diag(ones(1,m-1),-1) ;

% Construct Q
lambda =  true_parameters(5);
tau    =  true_parameters(4);
[X,Y]  =  meshgrid(1:m);
Q      =  tau^2 * (1+sqrt(3)*abs(X-Y)/lambda).*exp(-sqrt(3)*abs(X-Y)/lambda);
Q      =  (Q>1e-9).*Q;
clear X Y

% Constant
cons        = true_parameters(6);
c           = cons * ones(m,1);
identity    = eye(m);

%% Unconditional distribution of alpha (needed for  particle filter)
mu    = cons * ones(m,1);
tic
Sigma = reshape( ( eye(m*m)-kron(T,T) ) \ reshape(Q,m*m,1),m,m);
toc

%% Prefill stuff that will store our results
MAE_BF  = zeros(number_of_samples,t_final);
MAE_PF  = zeros(number_of_samples,t_final);
time_BF = zeros(number_of_samples,1);
time_PF = zeros(number_of_samples,1);
intervention = zeros(number_of_samples,1);

%% Do the simulations
for k = 1:number_of_samples

%% Prefill
eta           = zeros(m,t_final);
alpha         = zeros(m,t_final);
y             = zeros(m,t_final);

% Data generation
for t=1:t_final
    eta(:,t) = mvnrnd(zeros(m,1),Q)';
    if t==1
        alpha(:,t) = c; 
    else 
        alpha(:,t) =(identity-T) * c + T * alpha(:,t-1) + eta(:,t); 
    end
        y(:,t)  = poissrnd(exp(alpha(:,t)));
end

%% Run the Bellman filter
max_iterations=10^4;
now0=tic();
[a]=Bellman_filter1(y,true_parameters,max_iterations);
time_BF(k,1)=toc(now0);

%% Prefill some stuff for the particle filter
clear particles a_mean predicted_a_mean a_median predicted_a_median a_mean_prior_to_resampling eta weights sumweights weights_nor weights_stored resampling_id
a_mean                       = zeros(m,t_final);
a_mean_prior_to_resampling   = zeros(m,t_final);
a_median                     = zeros(m,t_final);
predicted_a_mean             = zeros(m,t_final);
predicted_a_median           = zeros(m,t_final);

%% Run the particle filter
now1=tic();
mu                         = cons * ones(m,1);
particles                  = mvnrnd(mu,Sigma,no_particles)';

%% Loop over time
for t=1:t_final
    %if mod(t,10)==0
    %disp([k,t])
    %end

    %% Prediction step
    eta         = mvnrnd(zeros(m,1),Q,no_particles)';
    particles   = (identity-T) * c + T * particles + eta;
    predicted_a_mean(:,t)   = mean(particles,2);
    predicted_a_median(:,t) = median(particles,2);

    %% Weights
    weights     = exp(logpdf(y(:,t), particles ));
    sumweights  = sum(weights);
    weights_nor = weights / sumweights;
    a_mean_prior_to_resampling(:,t) = sum(weights_nor .* particles,2);
    try
    resampling_id = randsample(no_particles,no_particles,true,weights_nor) ;
    catch
    resampling_id = randsample(no_particles,no_particles,true) ;
    intervention(k,1)=1;
    end
    particles     = particles(:,(resampling_id'));
    a_mean(:,t)   = mean(particles,2);
    a_median(:,t) = median(particles,2);
% Close loop over time
end
time_PF(k,1)=toc(now1);

%% Get performance measures
burn_in_period = 5;
MAE_BF(k,burn_in_period:end) = mean(abs(alpha(:,burn_in_period:end)-a(:,burn_in_period:end)).^1);
MAE_PF(k,burn_in_period:end) = mean(abs(alpha(:,burn_in_period:end)-a_median(:,burn_in_period:end)).^1);

%% Display the performance so far
if mod(k,10)==0
    disp(k)
disp([mean(MAE_BF(1:k,burn_in_period:end),'all');mean(MAE_PF(1:k,burn_in_period:end),'all')])
disp([mean(time_BF(1:k,burn_in_period:end)),mean(time_PF(1:k,burn_in_period:end))])
end

%% Close the loop over the replications
end













% %% Check the performance of the filter for a randomly picked series
% id=randi([1 m])
% figure
% plot(log(y(id,2:end)),'ko')
% hold on
% plot(alpha(id,2:end),'g')
% hold on
% plot(a(id,2:end),'r')