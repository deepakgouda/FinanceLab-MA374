#!/usr/bin/env python
# coding: utf-8

# In[ ]:


'''
                                     MA374 | Lab 08
                        Deepak Kumar Gouda | 160123054
'''


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [10, 5]


# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


def getStockPath(S0, r, mu, sig, n, varReduction=False):
    S_plus = np.zeros(n)
    S_plus[0] = S0
    if varReduction:
        S_minus = np.zeros(n)
        S_minus[0] = S0
    dt = 1.0/365
    W = np.random.randn(n)*dt
    for i in range(1, n):
        W = dt*np.random.randn()
        S_plus[i] = S_plus[i-1]*np.exp((r-0.5*(sig**2))*dt + sig*W)
        if varReduction:
            S_minus[i] = S_minus[i-1]*np.exp((r-0.5*(sig**2))*dt - sig*W)
    if varReduction:
        return S_plus, S_minus
    else:
        return S_plus


# In[4]:


def getAsianOptionPrice(S0, r, mu, sig, n, K, varReduction=False):
    dt = 1.0/365

    maxPaths = 500
    sumPayOff = 0
    step = 10
    X = range(10, maxPaths+1, step)
    price = np.zeros(len(X))

    for t, numPaths in enumerate(range(10, maxPaths+1, 10)):
        sumPayOff = 0
        for i in range(numPaths):
            if varReduction:
                S_plus, S_minus = getStockPath(S0, r, mu, sig, n, varReduction)
                V_plus = max(K - np.mean(S_plus), 0)
                V_minus = max(K - np.mean(S_minus), 0)
                sumPayOff = sumPayOff+(V_plus+V_minus)*0.5
            else:
                S = getStockPath(S0, r, mu, sig, n, varReduction)
                V = max(K - np.mean(S), 0)
                sumPayOff = sumPayOff+V
        price[t] = (np.exp(-r*(n)*dt)*sumPayOff/numPaths)
    return price


# In[5]:


S0 = 100.0
r = 0.05
mu = 0.1
sig = 0.2
n = 30
K = 110
maxPaths=500


# In[6]:


for K in [90, 105, 110]:
    # Without Variance reduction
    price = getAsianOptionPrice(S0, r, mu, sig, n, K, varReduction=False)

    X = np.arange(10, maxPaths+1, 10)
    Y = price
    plt.xlabel("Number of Paths")
    plt.ylabel("Option Price")
    plt.title("K = %d (without Variance Reduction)"%(K))
    plt.plot(X, Y)
    plt.show()

    # With Variance reduction
    price = getAsianOptionPrice(S0, r, mu, sig, n, K, varReduction=True)

    X = np.arange(10, maxPaths+1, 10)
    Y = price
    axes = plt.gca()
    if K is 110:
        axes.set_ylim([9.8375, 9.8425])
    plt.xlabel("Number of Paths")
    plt.ylabel("Option Price")
    plt.title("K = %d (with Variance Reduction)"%(K))
    plt.plot(X, Y)
    plt.show()


# In[7]:


S0 = 100.0
r = 0.05
mu = 0.1
sig = 0.2
n = 30
K = 110
maxPaths=500


# In[8]:


for r in np.arange(0.02, 0.11, 0.02):
    # Without Variance reduction
    price = getAsianOptionPrice(S0, r, mu, sig, n, K, varReduction=False)

    X = np.arange(10, maxPaths+1, 10)
    Y = price
    plt.xlabel("Number of Paths")
    plt.ylabel("Option Price")
    plt.title("r = %f (without Variance Reduction)"%(r))
    plt.plot(X, Y)
    plt.show()

    # With Variance reduction
    price = getAsianOptionPrice(S0, r, mu, sig, n, K, varReduction=True)

    X = np.arange(10, maxPaths+1, 10)
    Y = price
    plt.xlabel("Number of Paths")
    plt.ylabel("Option Price")
    plt.title("r = %f (with Variance Reduction)"%(r))
    plt.plot(X, Y)
    plt.show()


# In[9]:


S0 = 100.0
r = 0.05
mu = 0.1
sig = 0.2
n = 30
K = 110
maxPaths=500


# In[11]:


for sig in np.arange(0.1, 1.1, 0.2):
    # Without Variance reduction
    price = getAsianOptionPrice(S0, r, mu, sig, n, K, varReduction=False)

    X = np.arange(10, maxPaths+1, 10)
    Y = price
    plt.xlabel("Number of Paths")
    plt.ylabel("Option Price")
    plt.title("sig = %f (without Variance Reduction)"%(sig))
    plt.plot(X, Y)
    plt.show()

    # With Variance reduction
    price = getAsianOptionPrice(S0, r, mu, sig, n, K, varReduction=True)

    X = np.arange(10, maxPaths+1, 10)
    Y = price
    plt.xlabel("Number of Paths")
    plt.ylabel("Option Price")
    plt.title("sig = %f (with Variance Reduction)"%(sig))
    plt.plot(X, Y)
    plt.show()


# In[ ]:




