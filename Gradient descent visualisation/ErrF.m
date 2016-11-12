%Error function
%given a learning set, this function computes the cumulative squared error
%of the perceptron
% w1, w2 - weights, beta is the beta parameter of the sigmoid activation
% function f(s) = 1/(1+exp(-beta*s))
function ret = ErrF(w1, w2, beta)
learning_set = [1,1,0; -1,1,0 ; -1,0,0; 1,-1,1];
running_sum = 0;
    for n = 1:4
        vector = learning_set(n, :);
        z_i = vector(3); % correct value for the i-th vector from the learning set
        s_i = dot([w1, w2], [vector(1), vector(2)]); % excitation of the perceptron given the i-th vector from the learning set
        y_i = sigmoid(beta, s_i); % perceptron output
        squared_diff = (y_i - z_i)^2;
        running_sum = running_sum + squared_diff;
    end
    ret = 0.5 * running_sum;
end
% ta funkcja liczy skumulowany blad po wszystkich wektorach ustalonego ciagu uczacego