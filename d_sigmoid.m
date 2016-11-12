function ret = d_sigmoid(beta, x)
    ret = beta*sigmoid(beta,x)*(1 - sigmoid(beta,x));
end