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

%% Optimisation options (note that parallel = "true" )
clear options1 options2 options3
precision   = 6; 
options1    = optimset('fmincon');
options1    = optimset(options1, 'MaxFunEvals' , 10^precision, 'MaxIter' ,10^precision, 'TolFun' , 10^-precision, 'TolX' , 10^-precision,'display','final','UseParallel',true);

%% Load and display the data used by Catania in his JBES paper
load('SP500 data') 
figure
plot(y)

%% Initialise output for Bellman Filter
max_lags           = 10;
max_iterations     = 100;
BF_parameters      = zeros( 1 + max_lags , 4 + 1 + max_lags );
BF_standard_errors = zeros( 1 + max_lags , 4 + 1 + max_lags );
BF_LL              = zeros( 1 + max_lags , 1);
BF_exitflag        = zeros( 1 + max_lags , 1);
BF_time            = zeros( 1 + max_lags , 1);

%% Bellman filter optimisation routine (fmincon)

for m=0:max_lags
disp(m)
    if m==0
        start_vals = [median(y),0,0.98,0.25,-0.5,0];
        lb         = [-1 -1  0      0 -.99  0];
        ub         = [1   1  0.999  1 +0.99 0];
        nonlcon    = @nonlinearConstraint;
   else
        start_vals = [BF_parameters(m,1:4+m),0];
        lb         = [-1 -1  0      0 -0.99*ones(1,length(start_vals)-4)];
        ub         = [1   1  0.999  1  0.99*ones(1,length(start_vals)-4)];
        nonlcon    = @nonlinearConstraint;
    end
tic
[BF_parameters(1+m,1:4+1+max(1,m)),BF_LL(1+m,1),BF_exitflag(1+m,1)] = fmincon('NegativeLogLikelihood',start_vals,[],[],[],[],lb,ub,nonlcon,options1,y,max_iterations);
BF_time(1+m,1)=toc;
end

%% Compute standard errors and store in cell format to enable parallel computation
target_function = @(parameters) NegativeLogLikelihood(parameters,y,max_iterations);
output = cell(1,max_lags);
parfor m=1:max_lags
    %parfor m=1:max_lags 
    % use parfor if parallel computing is available
    disp(m)
    neg_hess   = num_hess( target_function , BF_parameters(1+m,1:4+1+max(m,1)), 10^-4 );
    output{m}  = sqrt(diag(inv(neg_hess)))';
    disp(output{m})
end

% Convert from cell format back into our format
for m=1:max_lags
  BF_standard_errors(1+m,1:4+1+max(1,m))  = output{m};
end

%% BIC analysis for Bellman filter
BF_BIC = -2 * (-BF_LL)/length(y) + log(length(y)) * (5:1:15)'/length(y);
[~,BIC_index]=min(BF_BIC);

figure
plot(BF_BIC)
figure
plot(-BF_LL)

