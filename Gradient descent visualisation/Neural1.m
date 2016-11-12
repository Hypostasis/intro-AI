% this script illustrates the learning curve for a perceptron trained with
% gradient descent
% initial weights' values:
w1 = 1;
w2 = 4;
% you may want to play with the beta parameter to see how it affects the
% shape of the error function

beta = 1.5;

%for high values, the algorithm will fail (get stuck on flat areas of the error
%function)
%%
error = ErrF(w1,w2, beta)
y_1 = sigmoid(1, dot([w1, w2], [1, -1]));
y_2 = sigmoid(1, dot([w1, w2], [-1, 0]));
y_3 = sigmoid(1, dot([w1, w2], [-1, 1]));
y_4 = sigmoid(1, dot([w1, w2], [1, 1]));
%
tolerance = 0.0001
error = ErrF(w1,w2, beta)
learning_set = [1,1,0; -1,1,0 ; -1,0,0; 1,-1,1];
h = 0.5;
t = 1;
W1 = [];
W2 = [];
Z = [];
iterations_counter = 0;
max_iterations = 20000;
while(error > tolerance && iterations_counter < max_iterations)
    %epoka
    %{
    for i= 1:4
        vector = learning_set(i, :);
        s = dot([w1, w2], [vector(1), vector(2)]);
        y = sigmoid(beta, s);
        
        w1 = w1 - h*(y - vector(3)) * d_sigmoid(beta, s)*vector(1);
        w2 = w2 - h*(y - vector(3)) * d_sigmoid(beta, s)*vector(2);
    end
    %}
    grad_1 = 0;
    grad_2 = 0;
    for i= 1:4
        vector = learning_set(i, :);
        s = dot([w1, w2], [vector(1), vector(2)]);
        y = sigmoid(beta, s);
        grad_1 = grad_1 + ((y - vector(3)) * d_sigmoid(beta, s)*vector(1));
        grad_2 = grad_2 + ((y - vector(3)) * d_sigmoid(beta, s)*vector(2));
        
    end
    w1 = w1 - h*grad_1;
    w2 = w2 - h*grad_2;
    error = ErrF(w1, w2, beta);
    w1
    w2
    W1 = [W1 w1];
    W2 = [W2 w2];
    Z = [Z error];
    iterations_counter = iterations_counter + 1;
end
W1;
W2;
p = plot3(W1, W2, Z)
p.Color = 'red';
y_1 = sigmoid(beta, dot([w1, w2], [1, -1]));
y_2 = sigmoid(beta, dot([w1, w2], [-1, 0]));
y_3 = sigmoid(beta, dot([w1, w2], [-1, 1]));
y_4 = sigmoid(beta, dot([w1, w2], [1, 1]));
hold on
X1 = -5:0.5:5;
X2 = -5:0.5:5;
Y = zeros(length(X1));


for i = 1:length(X1)
    for j = 1:length(X2)
        Y(i,j) = ErrF(X2(j), X1(i), beta);
    end
end

surf(X1,X2,Y);
hold off
