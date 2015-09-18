function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

steps = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30 ];
minError = Inf;

for i = 1:size(steps, 2)
	for j = 1:size(steps,2)
		model = svmTrain(X, y, steps(i), @(x1, x2) gaussianKernel(x1, x2, steps(j)));
		predictions = svmPredict(model, Xval);
		error = mean(double(predictions ~= yval));
		if error < minError
			minError = error;
			C = steps(i);
			sigma = steps(j);
		end
	end
end

end
