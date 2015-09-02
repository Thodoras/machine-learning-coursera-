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

% Update every value y(i) of y, into a logical vector, (1 x num_labels) which has value 1 only in the index y(i) 
y = eye(num_labels)(y,:);

% Forward propagation
a1 = X;
a2 = sigmoid([ones(m, 1), X] * Theta1');
a3 = sigmoid([ones(m, 1), a2] * Theta2');

% Compute cost
J = 1/m * sum(sum(-y.*log(a3) - (1 - y).*log(1-a3), 2));

% Remove temporarilly bias from Theta1 and Theta2.
NTheta1 = Theta1(:,2:end);
NTheta2 = Theta2(:,2:end);

% Compute regularized cost
J = J + (lambda / (2 * m)) * (sum(sum(NTheta1 .^ 2)) + sum(sum(NTheta2 .^ 2)));
         
% Compute gradients
% First compute di (errors) and Di.

d3 = a3 - y;
d2 = d3 * Theta2(:,2:end) .* sigmoidGradient([ones(m,1), X] * Theta1');

D1 = d2' * [ones(m,1), a1];
D2 = d3' * [ones(size(a2,1),1), a2];

Theta1_grad = (1/m) * D1;
Theta2_grad = (1/m) * D2;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end
