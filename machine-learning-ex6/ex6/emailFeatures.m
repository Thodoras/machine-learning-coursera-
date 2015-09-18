function x = emailFeatures(word_indices)
%EMAILFEATURES takes in a word_indices vector and produces a feature vector
%from the word indices
%   x = EMAILFEATURES(word_indices) takes in a word_indices vector and 
%   produces a feature vector from the word indices. 

% Total number of words in the dictionary
n = 1899;
m = length(word_indices);

% You need to return the following variables correctly.
x = zeros(n, 1);

for i = 1:m
	index = word_indices(i);
	if x(index) == 0
		x(index) = 1;
	end
end   

end
