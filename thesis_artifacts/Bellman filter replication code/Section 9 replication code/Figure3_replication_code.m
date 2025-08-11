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

%% Prefill some quantities

clear NegLL0 NegLL1 NegLL2 NegLL3
NegLL(:,1) =inf*ones(t_final,1);
NegLL0(:,1)=inf*ones(t_final,1); 
NegLL1(:,1)=inf*ones(t_final,1);
NegLL2(:,1)=inf*ones(t_final,1);
NegLL3(:,1)=inf*ones(t_final,1);

ML_parameters  = zeros(t_final,7);
ML_parameters1 = zeros(t_final,7);
ML_parameters2 = zeros(t_final,7);
ML_parameters3 = zeros(t_final,7);

exitflag1 = zeros(t_final,1);
exitflag2 = zeros(t_final,1);
exitflag3 = zeros(t_final,1);

chosen_routine = zeros(t_final,1);

prediction    = zeros(t_final,6);
truth         = zeros(t_final,6); 
MSE           = zeros(t_final,1);

%% This number determines overall precision
max_iterations = 10^3; % max number of BFGS iterations in Bellman filter 
precision      = 3; 
% precision is taken to be 4 for the results in the the paper, but this is
% time consuming to estimate
% For robustness, three optimisation routines are used at every
% time step, which is quite time consuming. 

%% Loop over time
for t=1:t_final
    disp(t)

%% Select data
data_expanding_window = clouddataobserved(:,1:t);

%% Set starting values using previous output when possible
if t==1
    starting_values = [0.3   0.3   0.3   1.5 8 -2 1/10];
else
    starting_values = ML_parameters(t-1,:);
end

%% Try at the starting point
NegLL0(t,1) = NegativeLogLikelihood2(starting_values,data_expanding_window,max_iterations);

%% Three optimisation routines for the overdispersed version of the model

% 1: fminsearch
clear options
options  =  optimset('fminsearch');
options  =  optimset(options , 'MaxFunEvals'  ,10^precision);
options  =  optimset(options , 'MaxIter'      ,10^precision);
options  =  optimset(options , 'TolFun'       ,10^(-precision));
options  =  optimset(options , 'TolX'         ,10^(-precision));
try
    [ML_parameters1(t,:),NegLL1(t,1),exitflag1(t,1)]=fminsearch('NegativeLogLikelihood2',starting_values,options,data_expanding_window,max_iterations);
    catch
end

% 2: fminunc
clear options
options  =  optimset('fminunc');
options  =  optimset(options , 'MaxFunEvals'  ,10^precision);
options  =  optimset(options , 'MaxIter'      ,10^precision);
options  =  optimset(options , 'TolFun'       ,10^(-precision));
options  =  optimset(options , 'TolX'         ,10^(-precision));
options  =  optimset(options , 'UseParallel',  true);
try
    [ML_parameters2(t,:),NegLL2(t,1),exitflag2(t,1)]=fminunc('NegativeLogLikelihood2',starting_values,options,data_expanding_window,max_iterations);
catch
end

% 3: fmincon
clear options
options  =  optimset('fmincon');
options  =  optimset(options , 'MaxFunEvals'  ,10^precision);
options  =  optimset(options , 'MaxIter'      ,10^precision);
options  =  optimset(options , 'TolFun'       ,10^(-precision));
options  =  optimset(options , 'TolX'         ,10^(-precision));
options  =  optimset(options , 'UseParallel',  true);
lb = [-1,-1,-1,0,0,-inf,0];
ub = [ 1, 1, 1,inf,15,inf,1];
try
    [ML_parameters3(t,:),NegLL3(t,1),exitflag3(t,1)]=fmincon('NegativeLogLikelihood2',starting_values,[],[],[],[],lb,ub,[],options,data_expanding_window,max_iterations);
catch
end

%% Decide which parameters to use
minNegLL = min([NegLL0(t,1),NegLL1(t,1),NegLL2(t,1),NegLL3(t,1)]);
    if minNegLL==NegLL0(t,1); NegLL(t,1) = NegLL0(t,1); ML_parameters(t,:) = starting_values;     chosen_routine(t,1) = 0; end
    if minNegLL==NegLL1(t,1); NegLL(t,1) = NegLL1(t,1); ML_parameters(t,:) = ML_parameters1(t,:); chosen_routine(t,1) = 1; end
    if minNegLL==NegLL2(t,1); NegLL(t,1) = NegLL2(t,1); ML_parameters(t,:) = ML_parameters2(t,:); chosen_routine(t,1) = 2; end
    if minNegLL==NegLL3(t,1); NegLL(t,1) = NegLL3(t,1); ML_parameters(t,:) = ML_parameters3(t,:); chosen_routine(t,1) = 3; end

 %% Run filter at these parameters
clear a1 Info1 predicted_a1 predicted_Info1 no_iterations1 delta_stored1
[a1,Info1,predicted_a1,predicted_Info1,no_iterations1,delta_stored1] = Bellman_filter2(data_expanding_window,ML_parameters(t,:),max_iterations);

 %% How well does this do on the test set?
 missing_data_indices = isnan(clouddataobserved(:,t));
 prediction(t,:)      = exp(a1(missing_data_indices,t))';
 truth(t,:)           = clouddataall(missing_data_indices,t)';
 MSE(t)               = mean((truth(t,:)-prediction(t,:)).^2);
 disp(mean(MSE))

end





%% Check the MSE again
for t=1:t_final
    disp(t)
    data_expanding_window = clouddataobserved(:,1:t);
    clear a1 Info1 predicted_a1 predicted_Info1 no_iterations1 delta_stored1 
    [a1,Info1,predicted_a1,predicted_Info1,no_iterations1,delta_stored1] = Bellman_filter2(data_expanding_window,ML_parameters(t,:),max_iterations);
    missing_data_indices = isnan(clouddataobserved(:,t));
    prediction(t,:)      = exp(a1(missing_data_indices,t))';
    truth(t,:)           = clouddataall(missing_data_indices,t)';
    MSE(t)               = mean((truth(t,:)-prediction(t,:)).^2);
 end

%% Get CRPS score (takes a while for 10^5 predictions)
prediction_vector = reshape( prediction, 480,1);
truth_vector      = reshape( truth, 480,1);
clear lambda probabilistic_prediction
probabilistic_prediction = zeros(480,10^5);
for j=1:480
    lambda                        = prediction_vector(j);
    probabilistic_prediction(j,:) = poissrnd( lambda , [1,10^5]);
end
crps(probabilistic_prediction,truth_vector)

%% Plot of the gamma's
figure('units','normalized','outerposition',[0 0 1 1])
axis([0 80 -0.05 0.4])
hold on
xticks([0 10 20 30 40 50 60 70 80])
yticks([-0.2 -0.1 0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0])
plot(ML_parameters(:,3),'linewidth',2,'LineStyle','-','Color','k')
hold on
plot(ML_parameters(:,1),'linewidth',2,'LineStyle',':','Color','k')
hold on
plot(ML_parameters(:,2),'linewidth',2,'LineStyle','--','Color','k')
hold on
legend('$\gamma_3$','$\gamma_1$','$\gamma_2$','Interpreter','latex')
set(gca,'FontName','Times','fontsize',30)
set(gca,'TickDir','out')
pbaspect([1 1 1])

%%
%ax = gca;
%exportgraphics(ax,'fig1.eps','Resolution',1000) 

%% Plot of the other parameters
ML_parameters(:,[4,5,7]) = abs( ML_parameters(:,[4,5,7]));

%%
figure('units','normalized','outerposition',[0 0 1 1])
axis([0 80 -10 12])
hold on
xticks([0 10 20 30 40 50 60 70 80])
plot(ML_parameters(:,5),'linewidth',2,'LineStyle',':','Color','k')
hold on
plot(ML_parameters(:,4),'linewidth',2,'LineStyle','-','Color','k')
hold on
plot(ML_parameters(:,7),'linewidth',2,'LineStyle','-.','Color','k')
hold on
plot(ML_parameters(:,6),'linewidth',2,'LineStyle','--','Color','k')
hold on
legend('$\lambda$','$\tau$','$\sigma_{\xi}$','$c$','Interpreter','latex')
set(gca,'FontName','Times','fontsize',30)
set(gca,'TickDir','out')
pbaspect([1 1 1])

%%
ax = gca;
exportgraphics(ax,'fig2.eps','Resolution',1000) 

