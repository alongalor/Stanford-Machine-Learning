function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables

% Initialize useful value
m = length(y); % number of training examples

% Variables to be returned
J = 0;
grad = zeros(size(theta));

% Cost function
J = (1/(m))*(sum((X*theta - y).^2)) + (lambda/(m))*sum(theta(2:end,:).^2);

% Gradient
grad(1,:) = (1/m)*X(:,1)'*((X*theta) - y);
grad(2:end,:) = (1/m)*X(:,2:end)'*((X*theta) - y) + (lambda/m)*theta(2:end,:);
grad = grad(:);

end
