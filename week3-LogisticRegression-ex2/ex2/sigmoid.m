function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   J = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).


% sigmoid == g(z) == 1 / [1 + e^-z]

%g = (1 + e^-z) .^-1               % only works for scalar, not matrix)
g = (((g .+ e) .^-z) .+1 ) .^-1    % matrix and scalar


% =============================================================

end
