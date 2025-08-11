function [y, a] = gendata(T, number_of_samples, ~)

params = truevals();
c       = params(1);
phi     = params(2);
sigman  = params(3);
k       = params(4);

%% Generate data

y       = zeros(number_of_samples, T);
a       = zeros(number_of_samples, T);

eta     = randn(number_of_samples, T);
for i = 1:number_of_samples

    for t=1:T
        if t==1
            a(i,t) = c/(1-phi);
        else
            a(i,t) = c + phi * a(i,t-1) + sigman * eta(i,t);
        end
    end
    y(i,:)  = gamrnd(repmat(k,1,T),exp(a(i,:)),1,T);
end


end

