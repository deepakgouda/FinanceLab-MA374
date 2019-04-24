#!/usr/bin/env python
# coding: utf-8

# In[1]:


from IPython.core.display import display, HTML


# In[2]:


display(HTML('<h1 style="text-align:center;">MA 374 | Assignment 9</h1>'))
display(HTML('<h2 style="text-align:center;">Deepak Kumar Gouda</h2>'))


# In[3]:


from pandas import read_csv, to_datetime
import numpy as np
from scipy.stats import norm


# In[4]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# In[5]:


get_ipython().run_line_magic('matplotlib', 'tk')


# In[6]:


fields=['Expiry', 'Strike Price', 'Put Price', 'Call Price']
orig_data = read_csv('NIFTYoptiondata.csv', usecols=fields, index_col=False)


# In[7]:


optionData = read_csv("NIFTYoptiondata.csv")
stockData = read_csv("./Data/nsedata1.csv")
optionData['Date2'] = to_datetime(optionData['Date'])
stockData['Date2'] = to_datetime(stockData['Date'])
stockData = stockData[['Date2','Close']]
data = optionData.merge(stockData,on='Date2')


# In[8]:


data.head()


# In[9]:


numSample = 1000
mask = np.random.randint(0, len(data), numSample)
data = data.loc[mask]


# In[10]:


data.head()


# In[11]:


len(data)


# In[12]:


import matplotlib.dates as mdates


# In[13]:


plot_data = orig_data[:numSample]


# In[14]:


def plotPrices(plot_data):
    dates = to_datetime(plot_data['Expiry'])
    x = to_datetime(dates)
    x = mdates.date2num(x)

    y = plot_data['Strike Price']
    z_call = plot_data['Call Price']
    z_put = plot_data['Put Price']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z_call, c='b', marker='.', label='Call Option')

    plt.xticks(x, data['Expiry'], rotation=90)
    ax.set_xlabel('Maturity Date')
    ax.set_ylabel('Strike Price')
    ax.set_zlabel('Option Prices')
    ax.legend()

    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z_put, c='r', marker='.', label='Put Option')

    plt.xticks(x, data['Expiry'], rotation=90)
    ax.set_xlabel('Maturity Date')
    ax.set_ylabel('Strike Price')
    ax.set_zlabel('Option Prices')
    ax.legend()

    plt.show()


# In[15]:


plotPrices(plot_data)


# In[16]:


display(HTML('<h3 style="text-align:center;">Maturity vs Strike Price vs Option Prices</h3>'))
display(HTML('<img src="Figure_3.png" alt="Drawing" style="width: 600px;"/>'))
display(HTML('<img src="Figure_4.png" alt="Drawing" style="width: 600px;"/>'))


# In[17]:


def getCall(S, K, r, t, sig):
    d1 = (np.log(S/K)+t*(r+(sig**2)/2))/(sig*(t**0.5))
    d2 = d1-sig*(t**0.5)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    C = S*Nd1 - K*np.exp(-r*t)*Nd2
    return C


# In[18]:


def getPut(S, K, r, t, sig):
    d1 = (np.log(S/K)+t*(r+(sig**2)/2))/(sig*(t**0.5))
    d2 = d1-sig*(t**0.5)
    Nd1 = norm.cdf(-d1)
    Nd2 = norm.cdf(-d2)
    P = K*np.exp(-r*t)*Nd2 - S*Nd1
    return P


# In[19]:


def f(Price, St, K, r, t, sig, option='Call'):
    if option is 'Call':
        return getCall(St, K, r, t, sig)-Price
    else:
        return getPut(St, K, r, t, sig)-Price


# In[20]:


def Secant(Price, St, K, r, t, option='Call'):
    x0 = 0.1
    x1 = 0.2
    
    tol = 0.00001
    num = 100
    alpha = 0.1
    for i in range(num):
        x2 = x1 - f(Price, St, K, r, t, x1, option)*(x1-x0)/(f(Price, St, K, r, t, x1, option)-f(Price, St, K, r, t, x0, option)+alpha)
        x0 = x1
        x1 = x2
#         print(x1, f(Price, St, K, r, t, x1, option))
        if abs(f(Price, St, K, r, t, x1, option)) < tol:
            break
    return x1


# In[21]:


from datetime import datetime


# In[22]:


num = len(data)
sig_c = np.zeros(num)
for i in range(num):
    St = data.iloc[-i]['Close']
    r = 0.05
    init_date=data.iloc[-i]['Date']
    exp_date=data.iloc[-i]['Expiry']
    
    date_format = "%d-%b-%Y"
    d0 = datetime.strptime(init_date, date_format)
    d1 = datetime.strptime(exp_date, date_format)
    t = (d1-d0).days/252
    K = data.iloc[-i]['Strike Price']
    P = data.iloc[-i]['Put Price']
    C = data.iloc[-i]['Call Price']
    
    sig_c[i] = Secant(C, St, K, r, t, 'Call')
    if abs(sig_c[i]) > 10:
        sig_c[i] = np.nan
#     if i%50 is 0:
#         print(str(i+1)+"/"+str(num))


# In[23]:


data.head()


# In[24]:


data['Volatility']=sig_c
data.drop(['Date2'], axis=1)
data.to_csv('result.csv', index=False)


# In[25]:


def plotVolatility(data):
    dates = to_datetime(data['Expiry'])
    x = to_datetime(dates)
    x = mdates.date2num(x)

    y = data['Strike Price']
    z = data['Volatility']

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.scatter(x, y, z, c='b', marker='.', label='Call Option')

    plt.xticks(x, data['Expiry'], rotation=90)
    ax.set_xlabel('Maturity Date')
    ax.set_ylabel('Strike Price')
    ax.set_zlabel('Volatility')
    ax.legend()
    plt.title('Maturity vs Strike Price vs Volatility')
    plt.show()


# In[26]:


plotVolatility(data)


# In[27]:


display(HTML('<h3 style="text-align:center;">Maturity vs Strike Price vs Volatility</h3>'))
display(HTML('<img src="Figure_1.png" alt="Drawing" style="width: 600px;"/>'))
display(HTML('<img src="Figure_2.png" alt="Drawing" style="width: 600px;"/>'))


# In[ ]:




