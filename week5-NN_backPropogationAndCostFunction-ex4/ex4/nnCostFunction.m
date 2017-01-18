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


% sh begin

% J: FORWARD PROPOGATION (hard coded for specific NN architecture):
  % CALCULATE h, classification output from X, given Theta1 and Theta2
  % X_plus_bias = [ones(m,1), X];

  % a2 = sigmoid(X_plus_bias * Theta1');
  % a2_plus_bias = [ones(m,1) , a2];

  % a3 = sigmoid(a2_plus_bias * Theta2');

  % h = a3;

% J: FORWARD PROPOGATION (generalized for L layers, using loops/arrays, see above):

  % create an array of Theta values, to take advantage of looping
  Thetas = {Theta1, Theta2};
  numLayers = size(Thetas, 2) + 1;

  % initialize input first layer1, == a1 before adding the bias term for a1
  g = X;

  for layer = 1:numLayers-1

    % (since don't need to store intermediate g's)(or z's, or a's):
    g = sigmoid( [ones(m,1), g] * Thetas{layer}' );

  end;

% result from our neural network:
h = g;

% J: NON-REGULARIZED COST, for X given Theta1, Theta2

 % J = (1/m) * sum( sum(-y .* log(h) - (1-y) .* log(1-h)) );

 %  RECODE y: vector of class-names into y_matrix of classification vectors
 %  RECODE y class labels (0..9) to vector representation of the classname
 %    (ie 4 -> [0 0 0 1 0 0 0 0 0 0])
 %    This turns y unit vector (5000 1) into matrix (5000 10)

Y = eye(num_labels)(y,:);

J =   (1/m) * sum( sum(-Y .* log(h) - (1-Y) .* log(1-h)) );

% J: REGULARIZATION term of cost function
regularization_cost = 0;
for layer = 1:numLayers-1;

  % this layer's Theta, without the bias term (regular weighted values only)
  Theta_no_bias = ( Thetas{layer} (:, 2:end) );

  regularization_cost += sum(sum( Theta_no_bias .^2 ));
end
regularization_cost *= lambda/(2*m);  % do this calc once instead of every loop

%  REGULARIZED COST
J = J + regularization_cost;


% sh end


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

% sh begin

Delta2 = Theta2_grad;
Delta1 = Theta1_grad;

% VECTORIZED VERSION OF BACKPROP/THETA_GRADIENT
% ... GENERALIZED TO L LAYERS, Looping through the NN Layers

%  These vars were  defined in the layers version of J, Cost Function
Thetas = {Theta1, Theta2};
numLayers = size(Thetas, 2) + 1;

%  1. FORWARD PROP

% initialize input first layer terms/neurons/nodes/source
g = X;                  % layer1, == a1 before adding the bias term for a1

for layer = 1: numLayers-1

  % input for this layer (without bias) is prev layer's output (or X, if layer1)
  a{layer} = g;

  %  add bias column to this activation layer
  a_plus_bias{layer} = [ ones(rows(a{layer}),1) , a{layer} ];

  z{layer+1} = a_plus_bias{layer} * Thetas{layer}';

  g = sigmoid(z{layer+1});

end  % forward prop through layers


% 2. BACKPROP on OUTPUT Layer:

delta{numLayers} = h - Y;

% 3. BACKPROP on HIDDEN Layer(s)

%  step backward through remaining layers, don't backprop on layer 1!
for layer = numLayers-1: -1 :2

  % convenience variable
  Theta_no_bias = ( Thetas{layer}(:, 2:end) );

  delta{layer} = delta{layer+1} * (Theta_no_bias) .* sigmoidGradient(z{layer});

end % Backprop for layers

% 4. Calculate Big Delta: Vector-multiply the deltas from m training samples..
for layer = 1:numLayers-1

  %  D2 (m x j3)' * (m x a2+bias) = (10 x m) * (m *  25+1) = (10 x  26)
  %  D1 (m x j2)' * (m x a1+bias) = (25 x m) * (m * 400+1) = (25 x 401)

  % Delta2 = delta3' * a2_plus_bias;
  % Delta1 = delta2' * a1_plus_bias;

  Delta{layer} = delta{layer+1}' * a_plus_bias{layer};


  % 5. UNREGULARIZED GRADIENT

  % Theta2_grad = Delta2/m;
  % Theta1_grad = Delta1/m;

  Theta_grad{layer} = Delta{layer} / m;

end  % calc Theta gradient, by looping through layers


% Part 3: Implement regularization with the cost function and gradients.


% REGULARIZATION TERM FOR GRADIENT in BackPropogation
for layer = 1:numLayers-1

  % Zero out this Theta's bias term
  % Theta_masked_bias = [ zeros(rows(Thetas{layer}),1) , Thetas{layer}(:,2:end) ];

  Theta_zeroed_bias = Thetas{layer};
  Theta_zeroed_bias(:,1) = 0;

  regularization_term_grad{layer} = (lambda/m) * Theta_zeroed_bias;

  Theta_grad{layer} += regularization_term_grad{layer};
end

% Move values to given variables
Theta1_grad = Theta_grad{1};
Theta2_grad = Theta_grad{2};

% sh end


% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];




end
