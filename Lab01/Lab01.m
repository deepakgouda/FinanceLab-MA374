s0=100;

K=100;

T=1;

M=100;

R=0.08;

sigma=0.2;

s=55:5:150;
k=55:5:150;
r=0.01:0.01:.2;
si=0.11:.01:.3;
m=55:5:150;

% Set 1 


call=[];

put=[];
    
for s0 = s
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
    u=exp((sigma*sqrt(t)));

    d=exp(-(sigma*sqrt(t)));

    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(1)
subplot(2,1,1);
plot(s,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs S(0)')
xlabel('S(0)')
ylabel('Call Option Price')

subplot(2,1,2);
plot(s,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs S(0)')
xlabel('S(0)')
ylabel('Put Option Price')

s0=100;

call=[];

put=[];

for K = k
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
    u=exp((sigma*sqrt(t)));

    d=exp(-(sigma*sqrt(t)));

    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(2)
subplot(2,1,1);
plot(k,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs K')
xlabel('K')
ylabel('Call Option Price')

subplot(2,1,2);
plot(k,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs K')
xlabel('K')
ylabel('Put Option Price')

K=100;

call=[];

put=[];

for R = r
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
    u=exp((sigma*sqrt(t)));

    d=exp(-(sigma*sqrt(t)));

    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(3)
subplot(2,1,1);
plot(r,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs r')
xlabel('r')
ylabel('Call Option Price')

subplot(2,1,2);
plot(r,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs r')
xlabel('r')
ylabel('Put Option Price')

R=0.08;

call=[];

put=[];

for sigma = si
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
    u=exp((sigma*sqrt(t)));

    d=exp(-(sigma*sqrt(t)));

    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(4)
subplot(2,1,1);
plot(si,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs sigma')
xlabel('sigma')
ylabel('Call Option Price')

subplot(2,1,2);
plot(si,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs sigma')
xlabel('sigma')
ylabel('Put Option Price')

call=[];

put=[];

sigma=0.2;

for M = m
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
    u=exp((sigma*sqrt(t)));

    d=exp(-(sigma*sqrt(t)));

    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(5)
subplot(2,1,1);
plot(m,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs M')
xlabel('M')
ylabel('Call Option Price')

subplot(2,1,2);
plot(m,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs M')
xlabel('M')
ylabel('Put Option Price')

K=95;
call=[];
put=[];


for M = m
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
    u=exp((sigma*sqrt(t)));

    d=exp(-(sigma*sqrt(t)));

    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(6)
subplot(2,1,1);
plot(m,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs M')
xlabel('M')
ylabel('Call Option Price')

subplot(2,1,2);
plot(m,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs M')
xlabel('M')
ylabel('Put Option Price')


K=105;
call=[];
put=[];


for M = m
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
    u=exp((sigma*sqrt(t)));

    d=exp(-(sigma*sqrt(t)));

    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(7)
subplot(2,1,1);
plot(m,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs M')
xlabel('M')
ylabel('Call Option Price')

subplot(2,1,2);
plot(m,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs M')
xlabel('M')
ylabel('Put Option Price')


% Set 2 

s0=100;

K=100;

T=1;

M=100;

R=0.08;

sigma=0.2;

s=55:5:150;
k=55:5:150;
r=0.01:0.01:.2;
si=0.11:.01:.3;
m=55:5:150;


call=[];

put=[];
    
for s0 = s
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
    u=exp((sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);

d=exp(-(sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);

    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(8)
subplot(2,1,1);
plot(s,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs S(0)')
xlabel('S(0)')
ylabel('Call Option Price')

subplot(2,1,2);
plot(s,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs S(0)')
xlabel('S(0)')
ylabel('Put Option Price')

s0=100;

call=[];

put=[];

for K = k
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
     u=exp((sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);

d=exp(-(sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);

    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(9)
subplot(2,1,1);
plot(k,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs K')
xlabel('K')
ylabel('Call Option Price')

subplot(2,1,2);
plot(k,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs K')
xlabel('K')
ylabel('Put Option Price')

K=100;

call=[];

put=[];

for R = r
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
    u=exp((sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);

d=exp(-(sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);


    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(10)
subplot(2,1,1);
plot(r,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs r')
xlabel('r')
ylabel('Call Option Price')

subplot(2,1,2);
plot(r,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs r')
xlabel('r')
ylabel('Put Option Price')

R=0.08;

call=[];

put=[];

for sigma = si
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
     u=exp((sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);

d=exp(-(sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);


    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(11)
subplot(2,1,1);
plot(si,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs sigma')
xlabel('sigma')
ylabel('Call Option Price')

subplot(2,1,2);
plot(si,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs sigma')
xlabel('sigma')
ylabel('Put Option Price')

call=[];

put=[];

sigma=0.2;

for M = m
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
   u=exp((sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);

d=exp(-(sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);

    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(12)
subplot(2,1,1);
plot(m,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs M')
xlabel('M')
ylabel('Call Option Price')

subplot(2,1,2);
plot(m,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs M')
xlabel('M')
ylabel('Put Option Price')

K=95;
call=[];
put=[];


for M = m
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
     u=exp((sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);

d=exp(-(sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);


    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(13)
subplot(2,1,1);
plot(m,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs M')
xlabel('M')
ylabel('Call Option Price')

subplot(2,1,2);
plot(m,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs M')
xlabel('M')
ylabel('Put Option Price')


K=105;
call=[];
put=[];


for M = m
     
    arr=zeros(M+1);
    crr=zeros(M+1);
    prr=zeros(M+1);
    t=T/M;
    arr(1,1)=s0;
   u=exp((sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);

d=exp(-(sigma*sqrt(t))+(R-(sigma*sigma)/2)*t);

    p=(exp(R*t)-d)/(u-d);
    
    for i=1:M
    for j=1:i
        arr(i+1,j)=arr(i,j)*(d);
        arr(i+1,j+1)=arr(i,j)*(u);
    end
    end
    
    for j=1:M+1
    crr(M+1,j)=max(0,arr(M+1,j)-K);
    prr(M+1,j)=max(0,K-arr(M+1,j));
    
    end
   
    for i=M:-1:1
    for j=1:i
        crr(i,j)=exp(-R*t)*(p*crr(i+1,j+1)+(1-p)*crr(i+1,j));
        prr(i,j)=exp(-R*t)*(p*prr(i+1,j+1)+(1-p)*prr(i+1,j));
    end
    end
    
    call=[call,crr(1,1)];

    put=[put,prr(1,1)];

end

figure(14)
subplot(2,1,1);
plot(m,call,'-o','Color',[0.5,0,0.9]);
title('Call Option price vs M')
xlabel('M')
ylabel('Call Option Price')

subplot(2,1,2);
plot(m,put,'-o','Color',[0.5,0.9,0]);
title('Put Option price vs M')
xlabel('M')
ylabel('Put Option Price')

