%% Clear and close
clear
close all

%% Automatically get the folder name where this file is stored 
file_name_full   = matlab.desktop.editor.getActiveFilename;
separator_id     = strfind(file_name_full,'/');
file_name_short  = file_name_full(max(separator_id)+1:end);
folder_name      = erase(file_name_full,file_name_short);

%% If the above doesn't work, you can also "hard code" the folder name
%folder_name = '/Users/Rutger-Jan/Dropbox (Erasmus Universiteit Rotterdam)/Section 8 replication code'

%% Tell Matlab to load the current folder
disp(folder_name)
cd(folder_name)

%% Remove previous things from the path 
% This generates some warnings that can be ignored
rmpath('LLt');
rmpath('NegB');
rmpath('Poisson');
rmpath('SCDe');
rmpath('SCDg');
rmpath('SCDw');
rmpath('SCg');
rmpath('SCt');
rmpath('SVg');
rmpath('SVt');

%% Decide which model to use. 
% Uncomment the model you want to use. 

modeltype = 'Poisson';
%modeltype = 'NegB';
%modeltype = 'SCDe';
%modeltype = 'SCDg';
%modeltype = 'SCDw';
%modeltype = 'SVg';
%modeltype = 'SVt';
%modeltype = 'SCg';
%modeltype = 'SCt';
%modeltype = 'LLt';

%% Add model to path
addpath(modeltype); 

%% Size of the simulation experiment

number_of_samples     = 10; 
% number of samples is 10^3 in the paper
t_final               = 5000; 
t_in_sample           = round(t_final/2);
window_size           = 1000; 
% window size is the number of observations used for the estimation of constant (hyper)parameters
t_out_of_sample       = t_final - t_in_sample;
true_param            = truevals();
start_vals            = truevals();

%% Generate data
tic
[y,alpha]=gendata(t_final,number_of_samples);
toc

%% Optimisation settings
options_a    = optimset('fminunc');
options_a    = optimset('display','off','TolFun',1e-5,'LargeScale','off','TolX',1e-5,'maxiter',10^4,'MaxFunEvals',10^4,'HessUpdate','bfgs','FinDiffType','central');
options_b    = optimset('fmincon');
options_b    = optimset('display','off','TolFun',1e-5,'LargeScale','off','TolX',1e-5,'maxiter',10^4,'MaxFunEvals',10^4,'HessUpdate','bfgs','FinDiffType','central');
options_c    = optimset('fminsearch');
options_c    = optimset('display','off','TolFun',1e-5,'LargeScale','off','TolX',1e-5,'maxiter',10^4,'MaxFunEvals',10^4,'HessUpdate','bfgs','FinDiffType','central');

%% Initialise everything with zeroes
BF_a0            = zeros(number_of_samples,t_final);
BF_a1            = zeros(number_of_samples,t_final);
BF_a2            = zeros(number_of_samples,t_final);
BF_a3            = zeros(number_of_samples,t_final);

BF_Info0         = zeros(number_of_samples,t_final);
BF_Info1         = zeros(number_of_samples,t_final);
BF_Info2         = zeros(number_of_samples,t_final);
BF_Info3         = zeros(number_of_samples,t_final);

BF_predicted_a0  = zeros(number_of_samples,t_final);
BF_predicted_a1  = zeros(number_of_samples,t_final);
BF_predicted_a2  = zeros(number_of_samples,t_final);

BF_predicted_Info0 = zeros(number_of_samples,t_final);
BF_predicted_Info1 = zeros(number_of_samples,t_final);
BF_predicted_Info2 = zeros(number_of_samples,t_final);

BF_ML_parameters_1    = zeros(number_of_samples,length(true_param));
BF_NegLogLikelihood_1 = inf*ones(number_of_samples,1);
BF_exitflag_1         = zeros(number_of_samples,1);

BF_ML_parameters_2a    = zeros(number_of_samples,length(true_param));
BF_NegLogLikelihood_2a = inf*ones(number_of_samples,1);
BF_exitflag_2a         = zeros(number_of_samples,1);

BF_ML_parameters_2b    = zeros(number_of_samples,length(true_param));
BF_NegLogLikelihood_2b = inf*ones(number_of_samples,1);
BF_exitflag_2b         = zeros(number_of_samples,1);

BF_ML_parameters_2c    = zeros(number_of_samples,length(true_param));
BF_NegLogLikelihood_2c = inf*ones(number_of_samples,1);
BF_exitflag_2c         = zeros(number_of_samples,1);

BF_ML_parameters_2    = zeros(number_of_samples,length(true_param));
BF_NegLogLikelihood_2 = inf*ones(number_of_samples,1);

BF_time_estimation    = zeros(number_of_samples,1);
BF_time_filtering     = zeros(number_of_samples,1);

chosen_optimisation_routine = nan * ones(number_of_samples,1);
max_iterations = 100;

%% Run estimations for Bellman filter
time_start=tic;
disp('Bellman filter started')
for i = 1:number_of_samples
    %parfor 1:number_of_samples
    % use parfor if parallel computing is available

%% Print progress
if mod(i,1) == 0
   disp(i)
end

%% Load the correct data
if or( isequal(modeltype,'SCg') , isequal(modeltype,'SCt') )
y_full              = y((2*i-1):(2*i) ,:  );
y_in_sample         = y((2*i-1):(2*i) , t_in_sample-window_size+1 : t_in_sample);
y_out_of_sample     = y((2*i-1):(2*i) , t_in_sample+1 : t_final);
else
y_full              = y(i, :);
y_in_sample         = y(i , t_in_sample-window_size+1 : t_in_sample);
y_out_of_sample     = y(i , t_in_sample+1 : t_final );
end

%% ML estimation (v1, based on out-of-sample data)
[BF_ML_parameters_1(i,:),BF_NegLogLikelihood_1(i,1),BF_exitflag_1(i,1)]=fminunc('BellmanfilterLL',start_vals,options_a,y_out_of_sample,max_iterations,modeltype);

%% ML estimation (v2, based on in-sample data)
now1=tic();
[BF_ML_parameters_2a(i,:),BF_NegLogLikelihood_2a(i,1),BF_exitflag_2a(i,1)]=fminunc('BellmanfilterLL',start_vals,options_a,y_in_sample,max_iterations,modeltype);
BF_time_estimation(i,1)=toc(now1);

%% Two more optimisation routines, just in case (these can be ``commented out'' for speed)
[BF_ML_parameters_2b(i,:),BF_NegLogLikelihood_2b(i,1),BF_exitflag_2b(i,1)]=fmincon('BellmanfilterLL',start_vals,[],[],[],[],[],[],[],options_b,y_in_sample,max_iterations,modeltype);
[BF_ML_parameters_2c(i,:),BF_NegLogLikelihood_2c(i,1),BF_exitflag_2c(i,1)]=fminsearch('BellmanfilterLL',start_vals,options_c,y_in_sample,max_iterations,modeltype);

%% Pick the best result of all three optimisation routines. 
bestLL = min([BF_NegLogLikelihood_2a(i,1),BF_NegLogLikelihood_2b(i,1),BF_NegLogLikelihood_2c(i,1)]);
BF_NegLogLikelihood_2 = bestLL;
if bestLL==BF_NegLogLikelihood_2a(i,1); BF_ML_parameters_2(i,:) = BF_ML_parameters_2a(i,:); chosen_optimisation_routine(i,1) = 1; end
if bestLL==BF_NegLogLikelihood_2b(i,1); BF_ML_parameters_2(i,:) = BF_ML_parameters_2b(i,:); chosen_optimisation_routine(i,1) = 2; end
if bestLL==BF_NegLogLikelihood_2c(i,1); BF_ML_parameters_2(i,:) = BF_ML_parameters_2c(i,:); chosen_optimisation_routine(i,1) = 3; end

%% Filtering at the true parameters (v0)
[BF_a0(i,:),BF_Info0(i,:),BF_predicted_a0(i,:),BF_predicted_Info0(i,:)]  = Bellmanfilter(y_full,start_vals,max_iterations,modeltype);

%% Filtering at the estimated (in-sample) parameters (v1)
[ BF_a1(i,:) , BF_Info1(i,:) , BF_predicted_a1(i,:) , BF_predicted_Info1(i,:)] = Bellmanfilter(y_full,BF_ML_parameters_1(i,:),max_iterations,modeltype);

%% Filtering at the estimated (out-of-sample) parameters (v2)
now2=tic();
[BF_a2(i,:), BF_Info2(i,:) , BF_predicted_a2(i,:) , BF_predicted_Info2(i,:)] = Bellmanfilter(y_full,BF_ML_parameters_2(i,:),max_iterations,modeltype);
BF_time_filtering(i,1)=toc(now2);

%% Smoothing at the estimated parameters 
[BF_a3(i,:), BF_Info3(i,:)] = Bellmansmoother(y_full,BF_ML_parameters_2(i,:),max_iterations,modeltype);

end
BF_total_time=toc(time_start);
disp(BF_total_time)
disp('Bellman filter done')

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% Particle filter

N = 100; % Number of particles: larger is better, but very computationally intensive 
% In the paper, I used 10^3 particles

%% Initialise everything with zeroes
PF_a0_mean       = zeros(number_of_samples,t_final);
PF_a0_median     = zeros(number_of_samples,t_final);

PF_a1_mean   = zeros(number_of_samples,t_final);
PF_a1_median = zeros(number_of_samples,t_final);

PF_a2_mean       = zeros(number_of_samples,t_final);
PF_a2_median     = zeros(number_of_samples,t_final);
PF_predictedlink_a2_mean= zeros(number_of_samples,t_final);

PF_ML_parameters_1    = zeros(number_of_samples,length(true_param));
PF_NegLogLikelihood_1 = zeros(number_of_samples,1);
PF_exitflag_1         = zeros(number_of_samples,1);

PF_ML_parameters_2    = zeros(number_of_samples,length(true_param));
PF_NegLogLikelihood_2 = zeros(number_of_samples,1);
PF_exitflag_2         = zeros(number_of_samples,1);

PF_time_estimation    = zeros(number_of_samples,1);
PF_time_filtering     = zeros(number_of_samples,1);

%% Run estimations for particle filter
disp('Particle filter started')
time_start=tic;
for i = 1:number_of_samples
%parfor i = 1:number_of_samples
    % use parfor if parallel computing is available

    %% Display progresss
if mod(i,1) == 0
   disp(i)
end

%% Load the correct data
if or( isequal(modeltype,'SCg') , isequal(modeltype,'SCt') )
    y_full              = y((2*i-1):(2*i) ,:  );
    y_in_sample         = y((2*i-1):(2*i) , t_in_sample-window_size+1 : t_in_sample);
    y_out_of_sample     = y((2*i-1):(2*i) , t_in_sample+1 : t_final);
else
    y_full              = y(i, :);
    y_in_sample         = y(i , t_in_sample-window_size+1 : t_in_sample );
    y_out_of_sample     = y(i , t_in_sample+1 : t_final );
end

%% Draw random variables
state_errors = randn(1,N,t_final);
unif_draws   = rand(1,t_final);
init_draws   = randn(1,N,1);  

%% Split random variables
state_errors_in_sample          = state_errors(1, :,t_in_sample - window_size+1        : t_in_sample  );
state_errors_out_of_sample      = state_errors(1, :,t_in_sample+1 : t_final );
unif_draws_in_sample            = unif_draws(1, t_in_sample - window_size+1        : t_in_sample  );
unif_draws_out_of_sample        = unif_draws(1, t_in_sample+1 : t_final );

%% Estimation
%[PF_ML_parameters_1(i,:),NegLogLikelihood_1b(i,1),exitflag_1b(i,1)] = fminunc('particlefilter',start_vals,options_a,y_out_of_sample,state_errors_out_of_sample,unif_draws_out_of_sample,init_draws);
tic
[PF_ML_parameters_2(i,:),PF_NegLogLikelihood_2(i,1),PF_exitflag_2(i,1)] = fminunc('particlefilter',start_vals,options_a,y_in_sample,state_errors_in_sample,unif_draws_in_sample,init_draws);
PF_time_estimation(i,1)=toc;

%% Filter
[~,PF_a0_mean(i,:),PF_a0_median(i,:),~] = particlefilter(start_vals,y_full,state_errors,unif_draws,init_draws);
%[~,PF_a1_mean(i,:),PF_a1_median(i,:),~] = particlefilter(PF_ML_parameters_1(i,:),y_full,state_errors,unif_draws,init_draws);
tic
[~,PF_a2_mean(i,:),PF_a2_median(i,:),PF_predictedlink_a2_mean(i,:)] = particlefilter(PF_ML_parameters_2(i,:),y_full,state_errors,unif_draws,init_draws);
PF_time_filtering(i,1)=toc;

%% close loop
end
PF_total_time=toc(time_start);
disp('Particle filter done')


%% Analysis of parameter estimates

% legend:
% true parameters
% average parameters based on Bellman filter 
% RMSE of Bellman filtered parameters
% average parameters based on particle filter
% RMSEs based on particle filter
format short
disp([true_param;
mean(BF_ML_parameters_2);
sqrt(mean(abs(BF_ML_parameters_2-true_param).^2));
mean(PF_ML_parameters_2); 
sqrt(mean(abs(PF_ML_parameters_2-true_param).^2))
])

%% Quality of filtered states (MAE)

% legend:
% MAE of Bellman filter, using the true parameters
% MAE of Bellman filter, parameters estimated based on evaluation period
% MAE of Bellman filter, parameters estimated based on estimation window
% MAE of Bellman smoother, parameters estimated based on estimation window
% MAE of particle filter, using the true parameters
% MAE of particle filter, parameters estimated based on estimation window

format short
disp([mean(mean(abs(alpha(:,t_in_sample+1:end)-BF_a0(:,t_in_sample+1:end))));
mean(mean(abs(alpha(:,t_in_sample+1:end)-BF_a1(:,t_in_sample+1:end))));
mean(mean(abs(alpha(:,t_in_sample+1:end)-BF_a2(:,t_in_sample+1:end))));
mean(mean(abs(alpha(:,t_in_sample+1:end)-BF_a3(:,t_in_sample+1:end))));
mean(mean(abs(alpha(:,t_in_sample+1:end)-PF_a0_median(:,t_in_sample+1:end))));
mean(mean(abs(alpha(:,t_in_sample+1:end)-PF_a2_median(:,t_in_sample+1:end))))])
