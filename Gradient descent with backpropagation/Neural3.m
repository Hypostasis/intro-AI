% this script runs a 2-layer neural network learning to approximate an
% arbitrary R^6 -> R^5 continuous and differentiable function with
% backpropagation and the momentum term
learning_set = [];
LS_len = 100;
for i = 1:LS_len
    input_vec = [rand()*2 - 2,rand()*2 - 2,rand()*2 - 2,rand()*2 - 2,rand()*2 - 2,rand()*2 - 2];
    output_vec = function65(input_vec(1),input_vec(2),input_vec(3),input_vec(4),input_vec(5),input_vec(6));
    learning_set = [learning_set; input_vec, output_vec];
end
%learning_set
%alfa 0.9, beta 0.2, h 0.1
alfa = 0.9;
beta = 0.2;
h = 0.1;
input = [1,1,1,1,1,1];
X_plot = [];

%initial weights
hidden = rand(8,6)
output = rand(5,8)
hidden_prev = hidden;
output_prev = output;

%hidden layer excitation
hidden_outputs = [];
for i = 1:8
    hidden_outputs = [hidden_outputs, sigmoid(beta, dot(input, hidden(i,:)))];
end
hidden_outputs

%output layer signals
outputs = []
for i = 1:5
    outputs = [outputs; dot(hidden_outputs,output(i,:))];
end
outputs

%learning epoch
for iterations = 1:1000
c_error = 0;
    %present learning set
    for i = 1:LS_len
    
        lsvec = learning_set(i, :);
        input = lsvec(1:6);
        d = lsvec(7:11);
        %hidden layer excitation
        hidden_s = [];
        hidden_outputs = [];
        for i = 1:8
            dotp = dot(input, hidden(i,:));
            hidden_s = [hidden_s, dotp];
            hidden_outputs = [hidden_outputs, sigmoid(beta, dotp)];
        end
        %compute outputs
        outputs = [];
        for i = 1:5
            outputs = [outputs, dot(hidden_outputs,output(i,:))];
        end
        %compute Q error
        error = 0;
        for j = 1:5
            error = error + (d(j)-outputs(j))^2;
        end
        %error
        c_error = c_error + error;
        
        %modify weights
        %output layer first
        delta_L = [0,0,0,0,0];
        for i = 1:5
            for j = 1:8
                %wij(t+1) = wij(t) + 2h* delta_L_i(t)* x_L_j(t)
                %delta_i = d_i(t) - y_i(t), memorize for backpropagation
                % xj(t) - output of the j-th neuron of the hidden layer
                delta_L(i) = (d(i) - outputs(i)) * d_sigmoid(beta, outputs(i));
                momentum_diff = output(i,j) - output_prev(i,j);
                output_prev(i,j) = output(i,j);
                output(i,j) = output(i,j) + 2*h * delta_L(i) * hidden_outputs(j) + alfa*momentum_diff;
            end
        end
        %hidden layer change
        for i = 1:8
            for j = 1:6
                %wij(t+1) = wij(t) + 2h * delta_1_i(t) * x_1_j(t)
                % delta_1_i(t) = eps_1_i(t) * d_sigmoid()
                eps_1_i = 0;
                for m = 1:5
                    eps_1_i = eps_1_i + delta_L(m) * output(m,i);
                end
                momentum_diff = hidden(i,j) - hidden_prev(i,j);
                hidden_prev(i,j) = hidden(i,j);
                hidden(i, j) = hidden(i,j) + 2 * h * eps_1_i * d_sigmoid(beta, hidden_s(i)) * input(j) + alfa*momentum_diff; 
                
                
            end
        end
        %end hidden layer change
        
    end
c_error
X_plot = [X_plot c_error];
iterations;
end
plot(1:iterations, X_plot)