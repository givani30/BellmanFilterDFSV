function [y, a] = gendata(T, number_of_samples, ~)

params = truevals();
c       = params(1);
phi     = params(2);
sigma_eta = params(3);
sigma   = params(4);
nu      = params(5);
%% Generate data

y       = zeros(number_of_samples, T);
a       = zeros(number_of_samples, T);

epsilon = trnd(nu,number_of_samples, T) * sqrt((nu - 2) / nu);
eta     = randn(number_of_samples, T);

for i = 1:number_of_samples

    for t=1:T
        if t==1
            a(i,t) = c/(1-phi);
        else
            a(i,t) = c + phi * a(i,t-1) + sigma_eta * eta(i,t);
        end
        y(i,t)     = a(i,t) + sigma * epsilon(i,t);
    end

end

end

