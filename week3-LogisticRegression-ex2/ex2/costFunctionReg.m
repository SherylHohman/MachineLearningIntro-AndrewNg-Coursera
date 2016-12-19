function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
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


% n = number of Features
% m is number of examples

% DON'T REGULARIZE theta_0 (aka theta_1 in Octave) (Cost AND Grad)
ignore_theta_0 = eye(size(theta, 1));
ignore_theta_0(1,1) = 0;
% matrix to ignore first theta term when computing regularization (lambda) term

%% SHOULDN'T THIS BE: h = sigmoid(theta' * X) ????
h = sigmoid(X * theta);

% COST:
logisticCost       = sum( (-1 * y .* log(h)) - ((1-y) .* log(1 - h)) );
regularizationCost = sum( ignore_theta_0 * (theta .^ 2) );

J = (1/m) * logisticCost + lambda/(2*m) * regularizationCost;


% GRADIENT:
% regularizer = (lambda/m) * ignore_theta_0 * theta;
% grad = (1/m) * sum ( (h - y) .* X)' + regularizer;

grad = (1/m)*sum((h - y).*X)' + (lambda/m)*ignore_theta_0*theta;


% SOMETHING IS WRONG HERE (GRAD):
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% I Should NOT be Transposing
%    sum ( (h - y) .* X)'

% This is what gives me the right answer.
% And how I obtained the correct dimensions to do the matrix math,
% But....

% Similarly, thetaT*X
% I've been doing as X * theta

% On both sections of this HW, and last week's HW

% Finally.. If something is wrong here,
% It's Also wrong in "constFunction.m"


% =============================================================

end
