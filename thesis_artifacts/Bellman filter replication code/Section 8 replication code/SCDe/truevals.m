function params = truevals(~)

%% Model constants (see Koopman, Lucas Scharth, Table 3)
c       = 0;
phi     = 0.98;
sigman  = 0.15;
%Q       = sigman^2; 
% Qinv= 1/Q;
params  = [c, phi, sigman];
end

