function [x_resample] = resample(x, weights_nor, u)

N           = size(weights_nor,1);

unif        = ((1:N)-1)/N + u/N;

x_resample  = zeros(N,1);
% Sort particles by size
[x,b]       = sort(x);
weights_nor = weights_nor(b);

lambda          = zeros(N,1);
lambda_0        = weights_nor(1) * 0.5;
lambda(N)       = weights_nor(N) * 0.5;
lambda(1:end-1) = 0.5 * (weights_nor(1:end-1) + weights_nor(2:end));

j   = 1;

r       = zeros(N,1);
u_star  = zeros(N,1);

s = lambda_0;
while j<=N && unif(j) <= s
    r(j)        = 0;
    u_star(j)   = (unif(j) - (s - lambda_0))/lambda_0; 
    j           = j+1;
end

for i = 1:N    
    s = s + lambda(i);
    
    while j<=N && unif(j) <= s
        r(j)        = i;
        u_star(j)   = (unif(j) - (s - lambda(i)))/lambda(i); 
        j           = j+1;        
    end
end

for j = 1:N
    ind = r(j);
    if ind == 0
        x_resample(j)   = x(1);
    elseif ind == N
        x_resample(j)   = x(N);
    else
        x_resample(j)   = (x(ind+1) - x(ind)) * u_star(j) + x(ind);
    end
    
end


end