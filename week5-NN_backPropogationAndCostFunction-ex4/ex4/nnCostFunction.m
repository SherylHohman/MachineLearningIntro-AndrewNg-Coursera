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

 %  RECODE y: vector of class-names into y_matrix of classification vectors
 %  RECODE y class labels (0..9) to vector representation of the classname
 %    (ie 4 -> [0 0 0 1 0 0 0 0 0 0])
 %    This turns y unit vector (5000 1) into matrix (5000 10)

 Y = eye(num_labels)(y,:);


% FORWARD PROPOGATION (hard code):
% CALCULATE h, classification output from X, given Theta1 and Theta2
% X_plus_bias = [ones(m,1), X];

% a2 = sigmoid(X_plus_bias * Theta1');
% a2_plus_bias = [ones(m,1) , a2];

% a3 = sigmoid(a2_plus_bias * Theta2');

% h = a3;

% FORWARD PROPOGATION (generalized for L layers, using loops/arrays, see above):
% CALCULATE h, classification output from X, given Theta1 and Theta2

% create an array of Theta values, to take advantage of looping
%   first layer does not have an input theta value
Thetas = {Theta1, Theta2};
numLayers = size(Thetas, 2) + 1;


% initialize input first layer terms/neurons/nodes/source
g = X;  % layer1, == a1 before adding the bias term for a1

for layer = 1:numLayers-1
  % a{layer} = [ones(m,1), g]; % add bias term to a{layer},
  %                             == g{layer-1} == g(z{layer-1}) from prev layer
  % z{layer+1} = a{layer} * Thetas{layer}'; % compute z for this layer
  % g{layer+1} = sigmoid( z{layer+1} );     % compute g(z) this layer: feeds layer a+1

  %  g{layer+1} = sigmoid( [ones(m,1), g{layer}] * Thetas{layer}' );

  % (since no need to store intermediate g's)(or z's, or a's):
  g = sigmoid( [ones(m,1), g] * Thetas{layer}' );
end;

% result from our neural network
h = g;

% J: NON-REGULARIZED COST, for X given Theta1, Theta2
% J = (1/m) * sum( sum(-y .* log(h) - (1-y) .* log(1-h)) );
J =   (1/m) * sum( sum(-Y .* log(h) - (1-Y) .* log(1-h)) );

% REGULARIZATION term of cost function
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


%
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
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the
%               first time.

% sh begin

Delta2 = Theta2_grad;
Delta1 = Theta1_grad;

% VECTORIZED VERSION OF BACKPROP/THETA_GRADIENT

  %  1. FORWARD PROP (on m_example)
                        %    j1 = 400 number of nodes layer1; j2=25; j3=10
                        %Theta1 % size:25 x 401 == j2 x (j1 + bias)
  a1 = X;               % size: 5000 x 400 ==  m x j1
  %  add bias column [ones(m,1), g]
  j1 = rows(a1);
  a1_plus_bias = [ ones(j1,1) , a1 ];    % size: 5k x 401 == m x j1+1
  z2 = a1_plus_bias * Theta1';  % size: 5k x  25 == m x j2
                                %               == (m x j1+bias)*(j2 x j1+bias)'
                                %               == (m x 401)    * (25 x 401)'
  g2 = sigmoid(z2);             % size: 5k x  25 == size(z2) == size(m x j2)

  a2 = g2;
  j2 = rows(a2);                     % m x j2   = 5k x 25
  a2_plus_bias = [(ones(j2,1)) a2];        % m x j2+1 = 5k x 26
  z3 = a2_plus_bias * Theta2';  %(m x j2+1)*(j3 x j2+1)' = (5k x j3) = (5k x 10)
  a3 = g3 = sigmoid(z3);        % m x j3   = 5k x 10

  h = a3;

  % 2. BACKPROP on OUTPUT Layer (m rows):
  %    delta3_k = (a3_k - y_k)
  %     (vector  :  (m x j3) - (m x hclasses) = (5000 x 10)-(5000x10) = 5000x10
  %     (oneRow i:  (1 x j3) - (1 x j3)       = (1 x j3) = (1x10)
  delta3 = a3 - Y;

  % 3. BACKPROP on HIDDEN Layer (on m rows):
  %    delta2_k = (Theta2_Transpose)(delta3) .* g'(z), where g'(z) is gradient
                          %Theta2  size:  j3 x (j2 + bias) == 10 x 26

  % size(Theta2_noBias')  % (j3 x j2)' =   (10  x 25)' = 25 x 10
  % size(delta3)--vector  %   m x j3   =  5000  x 10
  % size(delta3)--oneRow  %   1 x j3   =     1  x 10  (single training example)
  % size(z2)              %   m x j2   =     5k x 25
  % size(delta2)          %   m x j2   =     5k x 25 == (25x10) * (5kx10)
  %
  %  remove bias unit from Theta2  resulting size: j3 x j2 = 10 x 25
  Theta2_noBias = Theta2(:, 2:end);

  % this delta is for a matrix of all m examples (ie m=5k)
  % 5kx25= 5k x 10  *     (10x25)     .*    5k x 25
  delta2 = (delta3) * (Theta2_noBias) .* sigmoidGradient(z2);

  % 4. Calculate Big Delta: Vector-multiply the deltas from m training samples..

  %  Note that if loop thru training samples, m==1; if vectorize it, m==m=5k
  %  D2 (m x j3)' * (m x a2+bias) = (10 x m) * (m *  25+1) = (10 x  26)
  %  D1 (m x j2)' * (m x a1+bias) = (25 x m) * (m * 400+1) = (25 x 401)
  %  NEED a_WITH_BIAS !!
  Delta2 = delta3' * a2_plus_bias;
  Delta1 = delta2' * a1_plus_bias;


  % 5. UNREGULARIZED GRADIENT

  Theta2_grad = Delta2/m;
  Theta1_grad = Delta1/m;


% sh end


% UNVECTORIZED VERSION OF BACKPROP/THETA_GRADIENT
  % % loop thru the examples in our training set (m samples):
  % for m_example = 1:m

  %   %  1. FORWARD PROP (on m_example)
  %                         %    j1 = 400 number of nodes layer1; j2=25; j3=10
  %                         %Theta1 % size:25 x 401 == j2 x (j1 + bias)
  %   a1 = X(m_example, :);         % size: 1 x 400 ==  1 x j1
  %   a1_plus_bias = [1 a1];        % size: 1 x 401 ==  1 x j1+1
  %   z2 = a1_plus_bias * Theta1';  % size: 1 x  25 ==  1 x j2
  %                                 %               == (1 x j1+bias)*(j2 x j1+bias)'
  %                                 %               == (1 x 401)    * (25 x 401)'
  %   g2 = sigmoid(z2);             % size: 1 x  25 == size(z2) == size(1 x j2)

  %   a2 = g2;                      % 1 x j2   = 1 x 25
  %   a2_plus_bias = [1 a2];        % 1 x j2+1 = 1 x 26
  %   z3 = a2_plus_bias * Theta2';  % (1 x j2+1)*(j3 x j2+1)' = (1 x j3) = (1 x 10)
  %   a3 = g3 = sigmoid(z3);        % 1 x j3   = 1 x 10

  %   h = a3;

  %   % 2. BACKPROP on OUTPUT Layer (on row m_example, of m rows):
  %   %    delta3_k = (a3_k - y_k)
  %   %     (vector  :  (m x j3) - (m x hclasses) = (5000 x 10)-(5000x10) = 5000x10
  %   %     (oneRow i:  (1 x j3) - (1 x j3)       = (1 x j3) = (1x10)
  %   delta3 = a3 - Y(m_example, :);

  %   % 3. BACKPROP on HIDDEN Layer (on m_example):
  %   %    delta2_k = (Theta2_Transpose)(delta3) .* g'(z), where g'(z) is gradient
  %                           %Theta2  size:  j3 x (j2 + bias) == 10 x 26

  %   % size(Theta2_noBias')  % (j3 x j2)' =   (10 x 25)' = 25 x 10
  %   % size(delta3)--vector  %   m x j3   =  5000 x 10
  %   % size(delta3)--oneRow  %   1 x j3   =     1 x 10  (single training example)
  %   % size(z2)              %   1 x j2   =     1 x 25
  %   % size(delta2)          %   1 x j2   =     1 x 25 == (25x10) * (1x10)
  %   %
  %   %  remove bias unit from Theta2  resulting size: j3 x j2 = 10 x 25
  %   Theta2_noBias = Theta2(:, 2:end);

  %   % this delta is for a single training example (using for-loop) (ie m=1)
  %   %               Not a vector of all m examples (here m == 1)
  %   % 1x25 =  1 x 10  *     (10x25)     .*    1 x 25
  %   delta2 = (delta3) * (Theta2_noBias) .* sigmoidGradient(z2);

  %   % 4. Accumulate Big Delta: Sum Deltas as LOOP through all m training samples..

  %   %  Note that when looping thru training samples, m==1; if vectorize it, m==m=5k
  %   %  D2 (m x j3)' * (m x a2+bias) = (10 x m) * (m *  25+1) = (10 x  26)
  %   %  D1 (m x j2)' * (m x a1+bias) = (25 x m) * (m * 400+1) = (25 x 401)
  %   %  NEED a_WITH_BIAS !!
  %   Delta2 += (delta3' * a2_plus_bias);
  %   Delta1 += (delta2' * a1_plus_bias);

  % end

  % % 5. UNREGULARIZED GRADIENT

  % Theta2_grad = Delta2/m;
  % Theta1_grad = Delta1/m;


  % % sh end


%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% sh begin

% Zero-out Bias terms in Theta1, Theta2, since we don't Regularize on the bias term.  By masking that term, or zeroing it out, we can then write the formula in a vectorized manner, using a single formula, Rather than 1 for j=0, and another for j>-1  (In Octave the j=0 term is actually j=1, as Octave is 1-based, rather than 0-based.)

% REGULARIZATION TERM FOR GRADIENT in BackPropogation
Theta2_zeroed_bias = Theta2;
Theta2_zeroed_bias(:,1)  = 0;

Theta1_zeroed_bias = Theta1;
Theta1_zeroed_bias(:,1)  = 0;

regularization_term_grad2 = (lambda/m) * Theta2_zeroed_bias;
regularization_term_grad1 = (lambda/m) * Theta1_zeroed_bias;

%  add gradient Regularization term to the gradients
Theta1_grad += regularization_term_grad1;
Theta2_grad += regularization_term_grad2;

% sh end


% -----------------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];



end
