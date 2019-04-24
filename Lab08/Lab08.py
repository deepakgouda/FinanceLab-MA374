import numpy as np
from pandas import read_csv

from scipy.stats import norm

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def getS0(name='', market='BSE', index=False):
    fields=['Close']
    if index:
        data = read_csv('./Data/'+market.lower()+'data1.csv', usecols=fields, index_col=False)
    else:
        data = read_csv('./Data/'+market+'Data/'+name+'.csv', usecols=fields, index_col=False)
    return data.iloc[-1]['Close']

def getHistoricalVolatility(data):
    close = np.array(data['Close'])
    change = np.zeros(len(close)-1)
    for i in range(1, len(close)):
        change[i-1] = (close[i]-close[i-1])/close[i-1]
    historicalVolatility = np.std(change)*(252**0.5)
    return historicalVolatility

def getCall(S, K, r, t, sig):
    d1 = (np.log(S/K)+t*(r+(sig**2)/2))/(sig*(t**0.5))
    d2 = d1-sig*(t**0.5)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    C = S*Nd1 - K*np.exp(-r*t)*Nd2
    return C

def getPut(S, K, r, t, sig):
    d1 = (np.log(S/K)+t*(r+(sig**2)/2))/(sig*(t**0.5))
    d2 = d1-sig*(t**0.5)
    Nd1 = norm.cdf(-d1)
    Nd2 = norm.cdf(-d2)
    P = K*np.exp(-r*t)*Nd2 - S*Nd1
    return P

def model(name='', market='BSE', index=False, num_fig=0):
    fields=['Close']
    if index:
        data = read_csv('./Data/'+market.lower()+'data1.csv', usecols=fields, index_col=False)
    else:
        data = read_csv('./Data/'+market+'Data/'+name+'.csv', usecols=fields, index_col=False)
    
    if(index):
        print('\t\t'+market+'Index')
    else:
        print('\t\tStock : '+stock+' Market : '+market)
    ##################Q1####################
    lastMonth = data.iloc[len(data)-21:]
    sig = getHistoricalVolatility(lastMonth)
    print("Historical Volatility = ", sig)
    
    ##################Q2####################
    S0 = data.iloc[-1]['Close']
    A = np.arange(0.5, 1.6, 0.1)
    K = S0
    r = 0.05
    t = 126/252
    
    callPrice = np.zeros(len(A))
    putPrice = np.zeros(len(A))
    
    for i, a in enumerate(A):
        callPrice[i] = getCall(S0, a*K, r, t, sig)
        putPrice[i] = getPut(S0, a*K, r, t, sig)
    
    print('Call Price = ', end='')
    print(np.round(callPrice, 2))
    print('Put Price = ', end='')
    print(np.round(putPrice, 2))
    print('\n')
        
    ##################Q3###################
    S0 = data.iloc[-1]['Close']
    A = 1
    K = A*S0
    r = 0.05
    t = 126/252
    T = 21
    start = len(data)-T
    volatility = []
    callPrices = []
    putPrices = []
    
    while(start >= 0):
        monthlyData = data[start:-1]
        sig = getHistoricalVolatility(monthlyData)
        volatility.append(sig)
        C = getCall(S0, K, r, t, sig)
        P = getPut(S0, K, r, t, sig)
        callPrices.append(C)
        putPrices.append(P)
        start = start-T
        
    x = np.arange(1, len(volatility)+1)
    y = np.array(volatility)
    z_call = np.array(callPrices)
    z_put = np.array(putPrices)
    
    fig = plt.figure()
    ax = fig.add_subplot(111+num_fig, projection='3d')
    ax.scatter(x, y, z_call, c='b', marker='o', label='Call Option')
    ax.scatter(x, y, z_put, c='r', marker='o', label='Put Option')

    ax.set_xlabel('Time (in number of months)')
    ax.set_ylabel('Volatility')
    ax.set_zlabel('Option Prices')
    ax.legend()
    if index:
        plt.title(name+' Market = '+market)
    else:
        plt.title('Stock = '+name+' Market = '+market)
    plt.show()

stock_name=['ABB', 'AXISBANK', 'BALMLAWRIE', 'BHARTIARTL', 
          'CUMMINSIND', 'EMAMILTD', 'GODREJIND', 
          'HDFCBANK', 'HEROMOTOCO', 'HINDUNILVR', 
          'INFY', 'IOC', 'ITC', 'LUPIN', 'M&M', 
          'MAHABANK', 'NTPC', 'SBIN', 'SHREECEM', 'TCS']

market_name = ['BSE', 'NSE']

model('', 'BSE', index=True)
model('', 'NSE', index=True)


for i, market in enumerate(market_name):
    for j, stock in enumerate(stock_name):
        model(stock, market, num_fig=0)
