iter = 1000;
X = zeros(1, iter);
Y = zeros(1, faceiter);
for i = 1:iter
    point = model();
    X(i) = point(1);
    Y(i) = point(2);
end

scatter(X, Y)
title("Markowitz Efficient Frontier with "+iter+" points")
xlabel("Volatility")
ylabel("Return")

function point = model()
    mu = [0.1 0.2 0.15];
    cov = [0.005 -0.010 0.004; -0.010 0.040 -0.002; 0.004 -0.002 0.023];
    dim = length(cov);
    
    w = rand(1, 3);
    mu_v = dot(mu, w);
    sig_v = 0;
    for i = 1:dim
        sig_v = sig_v + ((w(i)*mu(i))^2)*(-cov(i, i) + 1);
    end

    for i = 1:dim
        for j = i+1:dim
            sig_v = sig_v + 2*(w(i)*w(j)*cov(i, j));
        end
    end
    point = [sig_v, mu_v];
end