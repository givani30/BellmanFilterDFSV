function [y, a] = gendata(t_final, number_of_samples, ~)

params    = truevals();
c         = params(1);
phi       = params(2);
sigma_eta = params(3);

%% Generate data
y       = zeros(2*number_of_samples, t_final);
a       = zeros(number_of_samples, t_final);

eta     = randn(number_of_samples, t_final);
for i = 1:number_of_samples

    for t=1:t_final
        if t==1
            a(i,t) = c/(1-phi);
        else
            a(i,t) = c + phi * a(i,t-1) + sigma_eta * eta(i,t);
        end
        rho                = link(a(i,t));
        Sigma              = [ 1 , rho ; rho , 1 ];
        y((2*i-1):(2*i),t) = transpose(mvnrnd([0;0],Sigma));
    end
end

end

