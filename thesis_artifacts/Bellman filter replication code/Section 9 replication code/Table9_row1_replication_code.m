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

%% Set the starting values for the optimisation for the standard model (sigma=0)
clear ML_parameters NegLL exitflag
%starting_values = [gamma1 gamma2 gamma3 tau lambda c];
starting_values = [0.33 0.33 0.33 1.8 7 -2];
max_iterations  = 10^3; % max number of iterations use din the Bellman filter
tic
NegativeLogLikelihood1(starting_values,clouddataobserved,max_iterations);
toc

%% Three optimisation routines to estimate the parameters for the standard model (sigma=0)
precision=4;
format bank

% 1: fminsearch
clear options
options  =  optimset('fminsearch');
options  =  optimset(options , 'MaxFunEvals'  ,10^precision);
options  =  optimset(options , 'MaxIter'      ,10^precision);
options  =  optimset(options , 'TolFun'       ,10^(-precision));
options  =  optimset(options , 'TolX'         ,10^(-precision));
[ML_parameters(1,:),NegLL(1,1),exitflag(1,1)]=fminsearch('NegativeLogLikelihood1',starting_values,options,clouddataobserved,max_iterations);

% 2: fminunc
clear options
options  =  optimset('fminunc');
options  =  optimset(options , 'MaxFunEvals'  ,10^precision);
options  =  optimset(options , 'MaxIter'      ,10^precision);
options  =  optimset(options , 'TolFun'       ,10^(-precision));
options  =  optimset(options , 'TolX'         ,10^(-precision));
options  =  optimset(options , 'UseParallel',  true);
[ML_parameters(2,:),NegLL(2,1),exitflag(2,1)]=fminunc('NegativeLogLikelihood1',starting_values,options,clouddataobserved,max_iterations);

% 3: fmincon
clear options
options  =  optimset('fmincon');
options  =  optimset(options , 'MaxFunEvals'  ,10^precision);
options  =  optimset(options , 'MaxIter'      ,10^precision);
options  =  optimset(options , 'TolFun'       ,10^(-precision));
options  =  optimset(options , 'TolX'         ,10^(-precision));
options  =  optimset(options , 'UseParallel',  true);
lb = [-1,-1,-1,0,0,-inf];
ub = [ 1, 1, 1,inf,10,inf];
[ML_parameters(3,:),NegLL(3,1),exitflag(3,1)]=fmincon('NegativeLogLikelihood1',starting_values,[],[],[],[],lb,ub,[],options,clouddataobserved,max_iterations);

%% Display results
format bank
disp([ML_parameters,NegLL/10^3,exitflag])

%% Decide which parameters to use
minNegLL = min([NegLL(1,1),NegLL(2,1),NegLL(3,1)]);
if minNegLL==NegLL(1,1); ML_parameters_best = ML_parameters(1,:); chosen_routine = 1; end
if minNegLL==NegLL(2,1); ML_parameters_best = ML_parameters(2,:); chosen_routine = 2; end
if minNegLL==NegLL(3,1); ML_parameters_best = ML_parameters(3,:); chosen_routine = 3; end
disp(chosen_routine)

%% Compute numerical standard errors (takes a while)
target_function = @(parameters) NegativeLogLikelihood1(parameters,clouddataobserved,max_iterations);
neg_hess        = num_hess( target_function , ML_parameters_best , 10^-3 );
se              = sqrt(diag(inv(neg_hess)))';

%% Round (to 3 decimal places) and display the result
ML_parameters_best = round(ML_parameters_best,3);
se = round(se,3);
disp([ML_parameters_best;se])

%% Filter the data with missing data at these parameters
tic
[a1] = Bellman_filter1(clouddataobserved,ML_parameters_best,max_iterations);
toc

%% Compute the MSE for predictions of the test set
prediction = exp(a1); % prediction of count is equal to the filtered intensity
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


