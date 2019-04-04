% s0 = 80.0:120.0;
s0 = 100.0;

k = 105.0;
% k = 90.0:110.0;

T = 5;

r = 0.05;
% r = 0.01:0.01:0.15;

sig = 0.3;
% sig = 0.1:0.1:0.5;

% m = 100;
m = 1:200;
model(s0, m, k, T, r, sig);

function model(s0, m, k, T, R, Sig)
    CallPricelist = [];
    PutPricelist = [];
    for z = 1:length(m)

        S0 = s0;
%         S0 = s0(z);
        
        K = k;
%         K = k(z);
               
        r = R;
%         r = R(z);
        
        sig = Sig;
%         sig = Sig(z);
        
%         M = m;
        M = m(z);        
        
        dt = T/M;

%         u = exp(sig*sqrt(dt));
%         d = exp(-sig*sqrt(dt));

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
                CallPayoff(i,j) = exp(-r*dt)*(p*CallPayoff(i,j+1) + (1-p)*CallPayoff(i+1,j+1));
                PutPayoff(i,j) = exp(-r*dt)*(p*PutPayoff(i,j+1) + (1-p)*PutPayoff(i+1,j+1));
            end
        end
        
        CallPricelist = [CallPricelist, CallPayoff(1, 1)];
        PutPricelist = [PutPricelist, PutPayoff(1, 1)];
    end

    X = m;
    Y1 = CallPricelist;
    Y2 = PutPricelist;
    
    plot(X, Y1);
    hold on
    plot(X, Y2);
    legend({"Call Option", "Put Option"}, 'Location', 'northeast')
    title("SET 2 : Variation of M 1 : 200");
    xlabel("S0")
    ylabel("Option price")
    hold off
end
