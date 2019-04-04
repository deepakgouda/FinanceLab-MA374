s0 = 100.0;
k = 105.0;
T = 1;
r = 0.08;
sig = 0.2;
m = 100;

% ------------------------S(0)----------------------

CallPricelist=[];
PutPricelist=[];
s0 = 80.0:120.0;
for S0 = s0
    Value = model(S0, m, k, T, r, sig);
    CallPricelist = [CallPricelist, Value(1)];
    PutPricelist = [PutPricelist, Value(2)];
end

X = s0;
Y1 = CallPricelist;
Y2 = PutPricelist;

figure(1)
plot(X, Y1, '-o', 'DisplayName', 'Call Option');
hold on
plot(X, Y2, '-o', 'DisplayName', 'Put Option');
legend({}, 'Location', 'northeast')
title('Variation of S(0) 80 : 120');
xlabel('S0')
ylabel('Option price')
hold off
s0 = 100.0;

% ------------------------K----------------------

CallPricelist=[];
PutPricelist=[];
k = 90:110;
for K = k
    Value = model(s0, m, K, T, r, sig);
    CallPricelist = [CallPricelist, Value(1)];
    PutPricelist = [PutPricelist, Value(2)];
end

X = k;
Y1 = CallPricelist;
Y2 = PutPricelist;

figure(2)
plot(X, Y1, '-o', 'DisplayName', 'Call Option');
hold on
plot(X, Y2, '-o', 'DisplayName', 'Put Option');
legend({}, 'Location', 'northeast')
title('Variation of K 90 : 110');
xlabel('S0')
ylabel('Option price')
hold off
k = 105.0;

% ------------------------r----------------------

CallPricelist=[];
PutPricelist=[];
R = 0.01:0.01:0.1;
for r = R
    Value = model(s0, m, k, T, r, sig);
    CallPricelist = [CallPricelist, Value(1)];
    PutPricelist = [PutPricelist, Value(2)];
end

X = R;
Y1 = CallPricelist;
Y2 = PutPricelist;

figure(3)
plot(X, Y1, '-o', 'DisplayName', 'Call Option');
hold on
plot(X, Y2, '-o', 'DisplayName', 'Put Option');
legend({}, 'Location', 'northeast')
title('Variation of R 0.01 : 0.1');
xlabel('S0')
ylabel('Option price')
hold off
r = 0.08;

% ------------------------sig----------------------

CallPricelist=[];
PutPricelist=[];
Sig = 0.1:0.1:0.5;
for sig = Sig
    Value = model(s0, m, k, T, r, sig);
    CallPricelist = [CallPricelist, Value(1)];
    PutPricelist = [PutPricelist, Value(2)];
end

X = Sig;
Y1 = CallPricelist;
Y2 = PutPricelist;

figure(4)
plot(X, Y1, '-o', 'DisplayName', 'Call Option');
hold on
plot(X, Y2, '-o', 'DisplayName', 'Put Option');
legend({}, 'Location', 'northeast')
title('Variation of sig 0.1 : 0.5');
xlabel('S0')
ylabel('Option price')
hold off
sig = 0.2;

% -------------------------M-------------------------
CallPricelist=[];
PutPricelist=[];
m = 1:200;
for M = m
    Value = model(s0, M, k, T, r, sig);
    CallPricelist = [CallPricelist, Value(1)];
    PutPricelist = [PutPricelist, Value(2)];
end

X = m;
Y1 = CallPricelist;
Y2 = PutPricelist;

figure(5)
plot(X, Y1);
hold on
plot(X, Y2);
legend({'Call Option', 'Put Option'}, 'Location', 'east')
title('Variation of M 1 : 200');
xlabel('S0')
ylabel('Option price')
hold off
m = 100;

% ------------------------function-----------------------
function Value = model(S0, M, K, T, r, sig)
    dt = T/M;

    u = exp(sig*sqrt(dt) + (r-0.5*(sig^2))*dt);
    d = exp(-sig*sqrt(dt) + (r-0.5*(sig^2))*dt);

    p = (exp(-r*dt) - d)/(u-d);
    dim = M+1;

    Stock = zeros(dim, dim);
    CallPayoff = zeros(dim, dim);
    PutPayoff = zeros(dim, dim);
    Stock(1,1) = S0;

    for j = 2:(dim)
        for i = 1:(j-1)
            Stock(i,j) = Stock(i,j-1)*u;
            Stock(i+1,j) = Stock(i,j-1)*d;
        end
    end

    for i = 1:dim
        CallPayoff(i,dim) = max(0, Stock(i,dim) - K);
        PutPayoff(i,dim) = max(0, K - Stock(i,dim));
    end

    for j = dim-1:-1:0
        for i = 1:j
            CallPayoff(i,j) = max(max(0, Stock(i, j)-K), exp(-r*dt)*(p*CallPayoff(i,j+1) + (1-p)*CallPayoff(i+1,j+1)));
            PutPayoff(i,j) = max(max(0, K-Stock(i, j)),exp(-r*dt)*(p*PutPayoff(i,j+1) + (1-p)*PutPayoff(i+1,j+1)));
        end
    end
    
    Value = [CallPayoff(1, 1), PutPayoff(1, 1)];
end
