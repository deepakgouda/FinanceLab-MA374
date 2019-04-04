s0=100;
T=1;
R=0.08;
sig=0.2;

m=5:5:25;
% m = 5;
call=[];

tic
    for M = m 
        dt=T/M;
        u=exp((sig*sqrt(dt))+(R-0.5*(sig^2))*dt);
        d=exp(-(sig*sqrt(dt))+(R-0.5*(sig^2))*dt);
        x=model(s0,T,M,R,sig,u,d);
        call=[call,x(1)];
    end
toc

figure(4)
plot(m,call,'-o');
title('Binomial Method : Call Option Price vs M')
xlabel('M')
ylabel('Option Price')

function value=model(s0,T,M,R,sig,u,d)

    dim = 2^(M+1)-1;
    Stock=zeros(1,dim);
    ma=zeros(1,dim);
    CallPayoff=zeros(1,dim);
    
    t=T/M;
    Stock(1)=s0;
    ma(1)=s0;
    p=(exp(R*t)-d)/(u-d);
    
    for i=1:2^(M)-1
        
      Stock(2*i)=Stock(i)*d;
      Stock(2*i+1)=Stock(i)*u;
      ma(2*i)=max(ma(i),Stock(2*i));
      ma(2*i+1)=max(ma(i),Stock(2*i+1));
    end
    
    for i=2^M:2^(M+1)-1
        CallPayoff(i)=max(0,ma(i)-Stock(i));
    end
    
    for i=2^(M)-1:-1:1
        CallPayoff(i)=exp(-R*t)*(p*CallPayoff(2*i)+(1-p)*CallPayoff(2*i+1));
    end
    value=[CallPayoff(1)];   
end