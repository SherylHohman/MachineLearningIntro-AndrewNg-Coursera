function [J, grad] = lrCostFunction(theta, X, y, lambda)
%LRCOSTFUNCTION Compute cost and gradient for logistic regression with
%regularization
%   J = LRCOSTFUNCTION(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations.
%
% Hint: When computing the gradient of the regularized cost function,
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta;
%           temp(1) = 0;   % because we don't add anything for j = 0
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% SH begin


% m is number of examples

% matrix to ignore bias term when computing regularization (lambda)
%   (theta_0, in zero-based systems) aka (theta_1, in Octave speak, 1-based)
terms_1_to_m = eye(size(theta, 1));
terms_1_to_m(1,1) = 0;

%% SHOULDN'T THIS BE: h = sigmoid(theta' * X) ????
z = X * theta;
h = sigmoid(z);

% COST for classification problems:
logisticCost       = sum( (-1 * y .* log(h)) - ((1-y) .* log(1 - h)) );
regularizationCost = sum( terms_1_to_m * (theta .^ 2) );

J = (1/m) * logisticCost + lambda/(2*m) * regularizationCost;


% GRADIENT: (dJ/dtheta_j):
% regularizer = (lambda/m) * terms_1_to_m * theta;
% grad = (1/m) * sum ( (h - y) .* X)' + regularizer;

grad = (1/m)*sum((h - y).*X)'  + (lambda/m)*(terms_1_to_m*theta);


% SH end


% =============================================================

grad = grad(:);

end
