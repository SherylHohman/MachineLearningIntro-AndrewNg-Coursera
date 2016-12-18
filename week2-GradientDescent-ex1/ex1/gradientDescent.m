function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

% sh convenience variable
alpha_div_m = alpha / m;

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta.
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
        h = X * theta;              % m x 2; 2 x 1 => m x 1
        diff = h - y;               % m x 1; m x 1 => m x 1
        diffX = X .* diff;          % m x 2; m x 1 ?> m x 2 ?
        sum_diffx = sum(diffX,1);   % m x 2 ?> 1 x 2 ? sums ea column
                                    % (ie 1: row returned; 2; column returned)
        theta = theta - alpha_div_m * sum_diffx';  % 2 x 1; 2 x 1 => 2 x 1

        %% Question: theta should be 2x1, not 1x2
        %% and ?? h = theta' * X (though thsat doesn't make sense)

        %% Question: why did I need to transpose sum_diffx?
        %% dshould I have done something differently, so I wouldn't need to?

    % ============================================================

    % Save the cost J in every iteration
    J_history(iter) = computeCost(X, y, theta);

end

end
