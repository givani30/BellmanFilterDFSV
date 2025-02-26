function [mu1,sigma1] = ito(mu, sigma, G, t, X)
%ITO Applies Itos rule to a symbolic function
%   [mu1,sigma1] = ito(mu, sigma, G, t, X) applies Itos rule to a symbolic function G
%   with respect to the variables t and X. It calculates the drift coefficient mu1
%   and the diffusion coefficient sigma1 using the given drift coefficient mu and
%   diffusion coefficient sigma.
%
%   Inputs:
%       - mu: Drift coefficient
%       - sigma: Diffusion coefficient
%       - G: Symbolic function
%       - t: Time variable
%       - X: State variable
%
%   Outputs:
%       - mu1: Updated drift coefficient
%       - sigma1: Updated diffusion coefficient
%
%   Example:
%       syms t X;
%       G = sin(t*X);
%       mu = 2*X;
%       sigma = X^2;
%       [mu1, sigma1] = ito(mu, sigma, G, t, X);
%
%   See also: diff

mu1 = mu * diff(G, X) + sigma^2/2 * diff(G, X, X) + diff(G, t);
sigma1 = diff(G,X) * sigma;
syms Y
mu1=
end