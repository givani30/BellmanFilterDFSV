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

%% Load data
load('clouddata.mat')
m       = size(clouddataall,1);
t_final = size(clouddataall,2);

%% Plot heatmap of the data
figure
heatmap(clouddataall)
title('all data')

figure
heatmap(clouddataobserved)
title('data with missing values')

%% OVERDISPERSED MODEL 

%% Set the starting values for the optimisation for the overdisperse model (sigma>0)
clear ML_parameters_new NegLL_new exitflag_new
%starting_values = [gamma1 gamma2 gamma3 tau lambda c sigma];
starting_values = [0.24 0.05 0.40 1.80 7.25 -4.26 0.05];
max_iterations  = 10^3; % max number of iterations use din the Bellman filter
format bank
tic
NegativeLogLikelihood2(starting_values,clouddataobserved,max_iterations);
toc

%% Three optimisation routines 
format short
clear ML_parameters_new NegLL_new exitflag_new
max_iterations=10^3;
precision=4;

% 1: fminsearch
clear options
options  =  optimset('fminsearch');
options  =  optimset(options , 'MaxFunEvals'  ,10^precision);
options  =  optimset(options , 'MaxIter'      ,10^precision);
options  =  optimset(options , 'TolFun'       ,10^-precision);
options  =  optimset(options , 'TolX'         ,10^-precision);
[ML_parameters_new(1,:),NegLL_new(1,1),exitflag_new(1,1)]=fminsearch('NegativeLogLikelihood2',starting_values,options,clouddataobserved,max_iterations);

%% 2: fminunc
starting_values = ML_parameters_new(1,:);
clear options
options  =  optimset('fminunc');
options  =  optimset(options , 'MaxFunEvals'  ,10^precision);
options  =  optimset(options , 'MaxIter'      ,10^precision);
options  =  optimset(options , 'TolFun'       ,10^-precision);
options  =  optimset(options , 'TolX'         ,10^-precision);
options  =  optimset(options , 'UseParallel',  true);
[ML_parameters_new(2,:),NegLL_new(2,1),exitflag_new(2,1)]=fminunc('NegativeLogLikelihood2',starting_values,options,clouddataobserved,max_iterations);

%% 3: fmincon
clear options
options  =  optimset('fmincon');
options  =  optimset(options , 'MaxFunEvals'  ,10^precision);
options  =  optimset(options , 'MaxIter'      ,10^precision);
options  =  optimset(options , 'TolFun'       ,10^-precision);
options  =  optimset(options , 'TolX'         ,10^-precision);
options  =  optimset(options , 'UseParallel',  true);
lb = [-1,-1,-1,0,0,-inf,0];
ub = [ 1, 1, 1,inf,30,inf,1];
[ML_parameters_new(3,:),NegLL_new(3,1),exitflag_new(3,1)]=fmincon('NegativeLogLikelihood2',starting_values,[],[],[],[],lb,ub,[],options,clouddataobserved,max_iterations);

%% Decide which parameters to use
minNegLL = min([NegLL_new(1,1),NegLL_new(2,1),NegLL_new(3,1)]);
if minNegLL==NegLL_new(1,1); ML_parameters_best = ML_parameters_new(1,:); chosen_routine = 1; end
if minNegLL==NegLL_new(2,1); ML_parameters_best = ML_parameters_new(2,:); chosen_routine = 2; end
if minNegLL==NegLL_new(3,1); ML_parameters_best = ML_parameters_new(3,:); chosen_routine = 3; end
disp(chosen_routine)

%% display the results
format short
disp([ML_parameters_new,NegLL_new/10^3,exitflag_new])

%% Compute numerical standard errors (this takes a while)
format short
target_function = @(parameters) NegativeLogLikelihood2(parameters,clouddataobserved,max_iterations);
neg_hess        = num_hess( target_function , ML_parameters_best , 10^(-3) );
se              = sqrt(diag(inv(neg_hess)))';

%% Round (to 3 decimal places) and display the result
ML_parameters_best = round(ML_parameters_best,3);
se = round(se,3);
format short
disp([ML_parameters_best;se])

%% Filter the data with missing data at these parameters
tic
[a2] = Bellman_filter2(clouddataobserved,ML_parameters_best,max_iterations);
toc

%% How well does this do on the test set?
prediction = exp(a2(1:60,:)); % prediction of count is equal to the filtered intensity
prediction = prediction(isnan(clouddataobserved)); % take the prediction only if the data is unobserved
truth      = clouddataall(isnan(clouddataobserved)); % take the truth to be the true count when the data is unobserved
MSE        = mean((truth-prediction).^2);
disp(MSE)

%% Get CRPS score (takes a while for 10^5 predictions)
clear lambda probabilistic_prediction
probabilistic_prediction = zeros(480,10^5);
for j=1:480
    lambda                        = prediction(j);
    probabilistic_prediction(j,:) = poissrnd( lambda , [1,10^5]);
end
crps(probabilistic_prediction,truth)

