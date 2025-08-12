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

%% Optimisation options
clear options
precision  = 6; % this is 6 in the paper
options    = optimset('fminsearch');
options    = optimset(options, 'MaxFunEvals' , 10^precision, 'MaxIter' ,10^precision, 'TolFun' , 10^-precision, 'TolX' , 10^-precision,'display','final');
% note: fminsearch is not gradient based, but parallel computing is not available for fminsearch

%% Load and display the data used by Catania in his JBES paper
load('SP500 data') 
figure
plot(y)
t_final = size(y,2);

%% Random draws for particle filter: these are fixed once and for all
N            = 5*10^3;
state_errors = randn(N,t_final);
unif_draws   = rand(t_final,1);
init_draws   = randn(N,1);    

%% Initialise output for PF
max_lags        = 10;
PF_parameters   = zeros( 1 + max_lags , 4 + 1 + max_lags );
PF_standard_errors = zeros( 1 + max_lags , 4 + 1 + max_lags );
PF_LL           = zeros( 1 + max_lags , 1);
PF_exitflag     = zeros( 1 + max_lags , 1);
time            = zeros( 1 + max_lags , 1);

%% Optimisation routine
for m=0:max_lags
disp(m)
    if m==0
        start_vals = [median(y),0,0.98,0.25,-0.7];
    else
        start_vals = [PF_parameters(m,1:4+m),0];
    end
tic
[PF_parameters(1+m,1:4+1+m),PF_LL(1+m,1),PF_exitflag(1+m,1)] = fminsearch('particlefilter',start_vals,options,y,state_errors,unif_draws,init_draws);
time(1+m,1)=toc;
disp( PF_parameters(1:1+m,1:4+1+m) )
end

%% Compute standard errors and store in cell format to enable parallel computation
target_function = @(parameters) particlefilter(parameters,y,state_errors,unif_draws,init_draws);
output = cell(1,max_lags);
%for m=1:max_lags
parfor m=1:max_lags    
% can use parfor if parallel computing is available
    disp(m)
    neg_hess   = num_hess( target_function , PF_parameters(1+m,1:4+1+max(m,1)), 10^-4 );
    output{m}  = sqrt(diag(inv(neg_hess)))';
    disp(output{m})
end

% Convert from cell format bank into our format
for m=1:max_lags
  PF_standard_errors(1+m,1:4+1+max(1,m))  = output{m};
end

%% Model selection and output
PF_BIC = -2 * (-PF_LL+length(y) * log(1/sqrt(2*pi)))/length(y) + log(length(y)) * (5:1:15)'/length(y);
[~,BIC_index]=min(PF_BIC);

figure
plot(PF_BIC) 

output= [PF_parameters,-PF_LL+length(y) * log(1/sqrt(2*pi)),PF_BIC];

