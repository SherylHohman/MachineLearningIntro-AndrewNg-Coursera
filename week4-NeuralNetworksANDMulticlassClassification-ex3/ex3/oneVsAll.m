function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda.
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with a large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
%
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

% sh start

% Set options for fminunc
    options = optimset('GradObj', 'on', 'MaxIter', 50);

% NOTES:
% m = size(X, 1) == number of training data
% n = size(X, 2) == number of parameters (for x and theta) in ea sample
% Adding ones to the X data matrix..
% X = [ones(m, 1) X]; % adds in the bias or x0 input of the neural network
% need to then use n+1 for number of params,
%   to account for the additional x0 bias element, which is always == 1

% train 10 classifiers (digits 0..9): determine theta, J for each digit
for digit_class = 1:num_labels
    % printf(' digit : %d ---\n', digit_class);

    initial_theta = zeros(n + 1, 1);
    % TODO: don't understand the notation "@(t)" and adding "t" as first param..
    f = @(t)lrCostFunction(t, X, (y == digit_class), lambda);
    [optim_theta] = fmincg (f, initial_theta, options);

    % printf('\n %d %d', size(optim_theta), size(all_theta));

    all_theta(digit_class:digit_class, :) = optim_theta;
end







% sh end


% =========================================================================


end
