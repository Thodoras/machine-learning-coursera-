function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

p = zeros(size(X, 1), 1);

X = [ones(m,1), X];

inner_X = sigmoid(X * Theta1');
inner_X = [ones(m,1), inner_X];

outer_X = sigmoid(inner_X * Theta2');

[dummy, p] = max(outer_X, [], 2);

end
