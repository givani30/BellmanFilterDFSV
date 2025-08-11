function [p] = pdf(y,a,~)
lambda  = exp(a);
p       = lambda .* exp(- lambda * y);
end