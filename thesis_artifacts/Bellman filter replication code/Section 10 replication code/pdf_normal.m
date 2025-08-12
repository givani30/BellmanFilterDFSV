function [p] = pdf_normal(y,mu,sigma)
p       = 1/(sqrt(2*pi)) * 1 ./ sigma .* exp(- 0.5 ./ sigma.^2 .* (y - mu).^2);
end