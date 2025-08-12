function [p] = pdf(y,a,~)
lambda  = exp(a);
p       = lambda.^y .* exp(-lambda) / gamma(y+1);
end