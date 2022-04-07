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
%
a_1 = [ones(m, 1) X];
z_2 = Theta1 * a_1';
a_2 = [ones(m, 1) sigmoid(z_2)'];
z_3 = Theta2 * a_2';
a_3 = sigmoid(z_3)';

for i = 1 : m
   for k = 1 : num_labels
      y_local = zeros(num_labels,1);
      y_local(y(i)) = 1;
      J = J + (-y_local' * (log(a_3(i,:)))' - (1 - y_local)'*(log(1 - a_3(i,:)))');
   endfor
endfor
J = J/(m * num_labels);

J_reg_input_layer = 0;
J_reg_hidden_layer = 0;
for j = 1 : hidden_layer_size
   for k = 2 : (input_layer_size + 1)
      J_reg_input_layer = J_reg_input_layer + Theta1(j, k).^2;
   endfor
endfor
for j = 1 : num_labels
   for k = 2 : (hidden_layer_size + 1)
      J_reg_hidden_layer = J_reg_hidden_layer + Theta2(j, k).^2;
   endfor
endfor
J = J + lambda * (J_reg_input_layer + J_reg_hidden_layer)/(2 * m);

% Part 2: Implement the backpropagation algorithm to compute the gradients
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
%
%Delta_1 = zeros(hidden_layer_size, 1);
%Delta_2 = zeros(num_labels, 1);
Delta_1 = zeros(size(Theta1));
Delta_2 = zeros(size(Theta2));
for t = 1 : m
   %initialization
   delta_3 = zeros(num_labels, 1);
   delta_2 = zeros(hidden_layer_size, 1);
   %Step 1: Feedforward to calculate the activations of each layer.
   a_1 = [1; X(t,:)'];
   z_2 = Theta1 * a_1;
   a_2 = [1; sigmoid(z_2)];
   z_3 = Theta2 * a_2;
   a_3 = sigmoid(z_3);
   %Step 2: Calculate error in the output layer.
   y_3 = zeros(num_labels,1);
   y_3(y(t)) = 1;
   delta_3 = a_3 - y_3; 
   %Step 3: Calculate error in the hidden layer.
   delta_2 = Theta2'(2:(hidden_layer_size + 1), :) * delta_3 .* sigmoidGradient(z_2);
   %Step 4: Accumulate the gradients.
   Delta_2 = Delta_2 + delta_3 * a_2';
   Delta_1 = Delta_1 + delta_2 * a_1';
endfor
   %Step 5: Obtain the gradient for the NN cost function.
   Theta1_grad = Delta_1 / m;
   Theta2_grad = Delta_2 / m;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


Theta1_reg = lambda * Theta1 / m;
Theta1_reg (:,1) = 0;
Theta2_reg = lambda * Theta2 / m;
Theta2_reg (:,1) = 0;

Theta1_grad = Theta1_grad + Theta1_reg;
Theta2_grad = Theta2_grad + Theta2_reg;



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
