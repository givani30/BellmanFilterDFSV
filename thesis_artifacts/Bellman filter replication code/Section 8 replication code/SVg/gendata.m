function [y, a] = gendata(T, number_of_samples, ~)

params = truevals();
c       = params(1);
phi     = params(2);
sigman  = params(3);
%% Generate data

y       = zeros(number_of_samples, T);
a       = zeros(number_of_samples, T);
eta     = zeros(1, T);
epsilon = zeros(1, T);

for i = 1:number_of_samples

    for t=1:T
        eta(1,t)            = sigman*randn;
        if t==1
            a(i,t) = c/(1-phi);
        else
            a(i,t) = c + phi * a(i,t-1) + eta(1,t);
        end
        epsilon(1,t)        = randn;
        y(i,t)              = sqrt(exp(a(i,t))) * epsilon(1,t);
    end

end

end

