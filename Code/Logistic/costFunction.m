function [J] = costFunction(theta, X, y, p)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

global ind_fair;global group_fair; global M; global lambda; global p_lvl;

% Initialize some useful values
n = length(y); % number of training examples
protected_levels = unique(p);

grad = zeros(size(theta));% m by 1 vector
h = sigmoid(X * theta);

J = -(1 / n) * sum( (y .* log(h)) + ((1 - y) .* log(1 - h)) ) ;

%% 
%RACE
if ind_fair == 1
    wx_2 = (repmat(X(p==protected_levels(1),:)*theta,1,p_lvl(2)) - ...
                repmat((X(p==protected_levels(2),:)*theta)',p_lvl(1),1)).^2;
            
    d_fun = penalty( repmat(y(p==protected_levels(1),:),1,p_lvl(2)) - ...
        repmat(y(p==protected_levels(2),:)',p_lvl(1),1));
    f = sum(sum(d_fun.* wx_2)); 
    J = J + 1/M*lambda * f;
end

if group_fair == 1
    wx_2 = (repmat(X(p==protected_levels(1),:)*theta,1,p_lvl(2)) - ...
        repmat((X(p==protected_levels(2),:)*theta)',p_lvl(1),1));

    d_fun = penalty( repmat(y(p==protected_levels(1),:),1,p_lvl(2)) - ...
        repmat(y(p==protected_levels(2),:)',p_lvl(1),1));
    f = sum(sum(d_fun.* wx_2));
    J = J + 1/M*lambda * f^2;
end
    

%grad = 1/n*X'*(h-y);
for i = 1 : size(theta, 1)
    grad(i) = (1 / n) * sum( (h - y) .* X(:, i) );
end




end
