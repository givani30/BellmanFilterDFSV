function params = truevals(~)

%% Model constants (see Koopman, Lucas Scharth, Table 3)
c       = 0.02;
phi     = 0.98;
sigman  = 0.1;
%Q       = sigman^2; 
% Qinv= 1/Q;

params  = [c, phi, sigman];
end

