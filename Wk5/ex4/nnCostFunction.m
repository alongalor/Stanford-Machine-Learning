function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% size(X) = (5000, 400)
% Add ones to the X data matrix
% size(a1) = (5000, 401)
a1 = [ones(m, 1) X];

% size(theta1) = (25, 401)
% size(a1') = (401, 5000)
% size(z2) = (25, 5000)
% size(a2) = (25, 5000)
z2 = Theta1*a1';
a2 = sigmoid(z2);

% size(a2) = (26, 5000)
a2 = [ones(1, size(a2,2)); a2];

% size(theta2) = (10, 26)
% size(a2) = (26, 5000)
% size(z3) = (10, 5000)
% size(a3) = (10, 5000)
z3 = Theta2*a2;
a3 = sigmoid(z3);

% creating a new y vector with 1's and 0's
yMod = zeros(num_labels,m)
for i=1:m,
    yMod(y(i),i) = 1;
end

% size yMod: 10x5000
% size a3: 10x5000
J = (1/m)*sum(sum(-yMod.*log(a3) - (1-yMod).*log(1-a3))) + (lambda/(2*m))*(sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)))

%Theta1_grad

%a1 = zeros(m, input_layer_size + 1)
%z2 = zeros(hidden_layer_size, m)
%a2 = zeros(hidden_layer_size + 1, m)
%z3 = zeros(num_labels, m)

%for t = 1:m,
    % ------- Part 1 ---------- %
    % size(a1(t,:)) = (1, 401)
    %a1(t,:) = [1 X(t,:)];
    % size(theta1) = (25, 401)
    % size(z2(:,t)) = (25, 1)
    % size(a2(:,t)) = (25, 1)
    %z2(:,t) = Theta1*a1(t,:)';
    %a2(:,t) = [1; sigmoid(z2(:,t))]
    % size(theta2) = (10, 26)
    % size(a2(:,t)) = (26, 1)
    % size(z3(:,t)) = (10, 1)
    % size(a3(:,t)) = (10, 1)
    %z3(:,t) = Theta2*a2(:,t);
    %a3(:,t) = sigmoid(z3(:,t));
Theta1(:,1) = 0  
Theta2(:,1) = 0 
    
d3 = a3 - yMod
d2 = Theta2(:,2:end)'*d3.*sigmoidGradient(z2)
Delta1 = d2*a1
Delta2 = d3*a2'

Theta1_grad = (1/m)*Delta1 + Theta1*(lambda/m)
Theta2_grad = (1/m)*Delta2 + Theta2*(lambda/m)

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
