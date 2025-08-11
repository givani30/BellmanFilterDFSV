function params = truevals(~)
%% Model constants 
c         = 0;
phi       = 0.98;
sigma_eta = 0.15;
%Q         = sigma_eta^2; 
sigma_noise = 0.45;
nu        = 3;
params    = [c, phi, sigma_eta, sigma_noise, nu];
end

