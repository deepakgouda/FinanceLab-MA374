#!/usr/bin/env python
# coding: utf-8

# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv


# In[17]:


get_ipython().run_line_magic('matplotlib', 'tk')


# In[18]:


from pandas import to_datetime
import pandas as pd


# In[19]:


def plotPrices(market='', stock='', period='Day', index=False):
    #####################Q1##########################
    fields=['Date', 'Open', 'Close']
    if index:
        data = read_csv('./'+market.lower()+'data1.csv', 
            usecols=fields, index_col=False)
    else:
        data = read_csv('./'+market.upper()+'Data/'+stock+'.csv', 
            usecols=fields, index_col=False)

    price = data

    if period is 'Day':
        price = data
    elif period is 'Week':
        data['Day'] = (to_datetime(data['Date'])).dt.day_name()
        price = data.loc[data['Day'] == 'Monday']
    else:
        price = data.groupby(pd.DatetimeIndex(data['Date']).to_period('M')).nth(0)
    X = np.arange(len(price))
    Y = np.array(price['Close'])

    if index:
        plt.title(market.upper()+" Index Data "+stock+" Period="+str(period))
    else:
        plt.title(market.upper()+" Stock Data "+stock+" Period="+str(period))
    plt.xlabel("Time in "+period+"s"+" since 1 January, 2014")
    plt.ylabel("Index Value")
    plt.plot(X, Y, label=period)
    plt.legend()


# In[20]:


def model(market='BSE', stock='', period='Day', index=False):
    if market is 'BSE':
        pad = 0
    else:
        pad = 3
    fields=['Date', 'Open', 'Close']
    if index:
        data = read_csv('./'+market.lower()+'data1.csv', 
            usecols=fields, index_col=False)
    else:
        data = read_csv('./'+market.upper()+'Data/'+stock+'.csv', 
            usecols=fields, index_col=False)

    price = data
    
    if period is 'Day':
        price = data
    elif period is 'Week':
        data['Day'] = (to_datetime(data['Date'])).dt.day_name()
        price = data.loc[data['Day'] == 'Monday']
    else:
        price = data.groupby(pd.DatetimeIndex(data['Date']).to_period('M')).nth(0)
    
    ret = np.array((price['Close']-price['Open'])/price['Open'])
    bins = 50

    #####################Q2##########################
    mean_ret = np.mean(ret)
    std_dev_ret = np.std(ret)
    norm_ret = (ret - mean_ret)/std_dev_ret   
    plt.subplot(2, 3, pad+1)
    
    if index:
        plt.title(market.upper()+" Index Data "+stock+" Period="+str(period))
    else:
        plt.title(market.upper()+" Stock Data "+stock+" Period="+str(period))
    plt.xlabel("Normal Return")
    plt.ylabel("Frequency")
    
    plt.hist(norm_ret, bins=bins, color='orange', 
                edgecolor='black', linewidth=0.3, density=True)

    mu, sig = 0, 1
    
    X = np.linspace(min(norm_ret), max(norm_ret), len(norm_ret))
    Y = (1/(2*np.pi*(sig**2))**0.5)*np.exp(-(X-mu)**2/(sig)**2)
    plt.plot(X, Y, color='blue', label = "Standard Normal")
    if period is 'Daily':
        plt.legend()

    #####################Q3##########################
    log_ret = np.log(1 + ret)
    mean_log_ret = np.mean(log_ret)
    std_dev_log_ret = np.std(log_ret)
    norm_log_ret = (log_ret - mean_log_ret)/std_dev_log_ret
    plt.subplot(2, 3, pad+2)

    if index:
        plt.title(market.upper()+" Index Data "+stock+" Period="+str(period))
    else:
        plt.title(market.upper()+" Stock Data "+stock+" Period="+str(period))
    plt.xlabel("Normalised Log Return")
    plt.ylabel("Frequency")

    plt.hist(norm_log_ret, bins=bins, color='orange',
         edgecolor='black', linewidth=0.3, density=True)

    mu, sig = 0, 1
    
    X = np.linspace(min(norm_log_ret), max(norm_log_ret), len(norm_log_ret))
    Y = (1/(2*np.pi*(sig**2))**0.5)*np.exp(-(X-mu)**2/(sig)**2)
    plt.plot(X, Y, color='blue', label = "Standard Normal")
    if period is 'Day':
        plt.legend()

    #######################Q4########################
    if period is 'Day':
        initial_ret = log_ret[:987]
        mu = np.sum(initial_ret)/len(initial_ret)/240
        sig = np.std(initial_ret)

        n = len(log_ret) - 987
        phi = np.random.normal(0, 1, n)
        W = np.zeros(n)
        W[0] = 0
        for i in range(1, n):
            W[i] = W[i-1]+phi[i]

        S = np.zeros(n)
        S[0] = price.iloc[987]['Close']
        for i in range(1, n):
            S[i] = S[0]*np.exp(sig*W[i]+(log_ret[987+i]-0.5*(sig**2))*i/240)

        S = np.reshape(S, (len(S), 1))
        actPrice = np.array(price[fields[1]])
        predPrice = actPrice[:987]
        predPrice = np.reshape(predPrice, (len(predPrice), 1))
        predPrice = np.vstack((predPrice, S))
        Y1 = predPrice
        Y2 = actPrice
        X = np.arange(len(Y1))
        plt.subplot(2, 3, pad+3)
        plt.title('Actual Price vs Calculated Price')
        plt.plot(X, Y1, color='blue', label='Predicted Price')
        plt.plot(X, Y2, color='red', label='Actual Price')
        plt.xlabel('Time')
        plt.ylabel('Stock Price')
        plt.legend()


# In[21]:


stock_name=['ABB', 'AXISBANK', 'BALMLAWRIE', 'BHARTIARTL', 
          'CUMMINSIND', 'EMAMILTD', 'GODREJIND', 
          'HDFCBANK', 'HEROMOTOCO', 'HINDUNILVR', 
          'INFY', 'IOC', 'ITC', 'LUPIN', 'M&M', 
          'MAHABANK', 'NTPC', 'SBIN', 'SHREECEM', 'TCS']
periodicity = ['Day', 'Week', 'Month']


# In[24]:


f = 1
for name in stock_name:
    ax = plt.figure(f, figsize=(15.0, 8.0))
    for i in range(len(periodicity)):
        plt.subplot(2, len(periodicity), i+1)
        plotPrices(market='BSE', stock=name, period=periodicity[i], index=False)

    for i in range(len(periodicity)):
        plt.subplot(2, len(periodicity), 4+i)
        plotPrices(market='NSE', stock=name, period=periodicity[i], index=False)
    ax.savefig('./Plots/'+name+'_1.png')
    ax.show()
    f = f+1


# In[25]:


f = 1
for name in stock_name:
    ax = plt.figure(f, figsize=(15.0, 8.0))
    for i in range(len(periodicity)):
        model(market='BSE', stock=name, period=periodicity[i], index=False)
        model(market='NSE', stock=name, period=periodicity[i], index=False)
    ax.savefig('./Plots/'+name+'_2.png')
    ax.show()
    f = f+1


# In[ ]:




