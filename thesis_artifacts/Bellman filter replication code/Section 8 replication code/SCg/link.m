function [output] = link(a)
output      = 2 * (1 ./ (1 + exp(-a))- 0.5);
end

