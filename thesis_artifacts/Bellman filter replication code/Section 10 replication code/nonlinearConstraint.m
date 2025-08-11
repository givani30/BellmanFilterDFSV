function [c,ceq] = nonlinearConstraint(x, ~, ~,~,~,~,~)
m = length(x) - 4;
if m == 1
    c = x(5)^2 - 1;
elseif m > 1
    c = sum(x(5:end).^2) - 1;
else 
    c = [];
end

ceq = [];
end

