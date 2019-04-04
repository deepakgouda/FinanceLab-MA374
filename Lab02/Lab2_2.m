s0=100;
T=1;
M=10;
R=0.08;
sig=0.2;

s=55:5:150;
r=0.01:0.01:.2;
si=0.11:.01:.3;
m=2:20;

dt=T/M;
u=exp((sig*sqrt(dt)));
d=exp(-(sig*sqrt(dt)));

call=[];
put=[];
for s0 = s
    dt=T/M;
    
    x=model(s0,T,M,R,sig,u,d);
    call=[call,x(1)];
    put=[put,x(2)];
    
end

figure(1)
plot(s,call,'-o');
hold on
title('Option Price vs S(0)')
xlabel('S(0)')
ylabel('Option Price')

plot(s,put,'-o');
legend({"Put Option", "Call Option"}, 'Location', 'northeast')
title('Option Price vs S(0)')
xlabel('S(0)')
ylabel('Option Price')
hold off

call=[];
put=[];

s0=100;
call=[];
put=[];

for R = r    
    dt=T/M;
    x=model(s0,T,M,R,sig,u,d);
    call=[call,x(1)];
    put=[put,x(2)];

end

figure(2)
plot(r,call,'-o');
hold on
title('Option Price vs r')
xlabel('r')
ylabel('Option Price')

plot(r,put,'-o');
title('Option Price vs r')
xlabel('r')
ylabel('Option Price')
legend({"Call Option", "Put Option"}, 'Location', 'northeast')
hold off

R=0.08;
call=[];
put=[];
for sig = si
    dt=T/M;
    u=exp((sig*sqrt(dt)));
    d=exp(-(sig*sqrt(dt)));
    x=model(s0,T,M,R,sig,u,d);
    call=[call,x(1)];
    put=[put,x(2)];
end

figure(3)
plot(si,call,'-o');
hold on
title('Option Price vs sig')
xlabel('sig')
ylabel('Option Price')

plot(si,put,'-o');
title('Option Price vs sig')
xlabel('sig')
ylabel('Option Price')
legend({"Call Option", "Put Option"}, 'Location', 'northeast')
hold off

call=[];
put=[];

sig=0.2;
R=0.08;
T=1;
s0=100;
for M = m 
    dt=T/M;
    u=exp((sig*sqrt(dt)));
    d=exp(-(sig*sqrt(dt)));
    x=model(s0,T,M,R,sig,u,d);
    call=[call,x(1)];
    put=[put,x(2)];
end

figure(4)
plot(m,call,'-o');
hold on
title('Option Price vs M')
xlabel('M')
ylabel('Option Price')

plot(m,put,'-o');
title('Option Price vs M')
xlabel('M')
ylabel('Option Price')
legend({"Call Option", "Put Option"}, 'Location', 'northeast')
hold off

function value=model(s0,T,M,R,sig,u,d)

    dim = 2^(M+1)-1;
    Stock=zeros(1,dim);
    ma=zeros(1,dim);
    mi=zeros(1,dim);
    CallPayoff=zeros(1,dim);
    PutPayoff=zeros(1,dim);
    
    t=T/M;
    Stock(1)=s0;
    ma(1)=s0;
    mi(1)=s0;
    p=(exp(R*t)-d)/(u-d);
    
    for i=1:2^(M)-1
        
      Stock(2*i)=Stock(i)*d;
      Stock(2*i+1)=Stock(i)*u;
      ma(2*i)=max(ma(i),Stock(2*i));
      mi(2*i)=min(mi(i),Stock(2*i));
      ma(2*i+1)=max(ma(i),Stock(2*i+1));
      mi(2*i+1)=min(mi(i),Stock(2*i+1));
    end
    
    for i=2^M:2^(M+1)-1
        CallPayoff(i)=max(0,Stock(i)-mi(i));
        PutPayoff(i)=max(0,ma(i)-Stock(i));
    end
    
    for i=2^(M)-1:-1:1
        CallPayoff(i)=exp(-R*t)*(p*CallPayoff(2*i)+(1-p)*CallPayoff(2*i+1));
        PutPayoff(i)=exp(-R*t)*(p*PutPayoff(2*i)+(1-p)*PutPayoff(2*i+1));
    end
    
    value=[CallPayoff(1),PutPayoff(1)];
    
end