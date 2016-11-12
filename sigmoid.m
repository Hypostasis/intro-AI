function ret = sigmoid(beta, x)
    ret = 1 ./ (1 + exp(-1 * beta .* x));
end