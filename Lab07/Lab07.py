#!/usr/bin/env python
# coding: utf-8

# In[19]:


import numpy as np
from scipy.stats import norm


# In[20]:


from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


# In[21]:


from tabulate import tabulate


# In[22]:


get_ipython().run_line_magic('matplotlib', 'tk')


# In[23]:


def getCall(S, t, K, r, sig, T=1):
    t = T-t
    d1 = (np.log(S/K)+t*(r+(sig**2)/2))/(sig*(t**0.5))
    d2 = d1-sig*(t**0.5)
    Nd1 = norm.cdf(d1)
    Nd2 = norm.cdf(d2)
    C = S*Nd1 - K*np.exp(-r*t)*Nd2
    return C


# In[24]:


def getPut(S, t, K, r, sig, T=1):
    t = T-t
    d1 = (np.log(S/K)+t*(r+(sig**2)/2))/(sig*(t**0.5))
    d2 = d1-sig*(t**0.5)
    Nd1 = norm.cdf(-d1)
    Nd2 = norm.cdf(-d2)
    P = K*np.exp(-r*t)*Nd2 - S*Nd1
    return P


# In[25]:


T = 1
K = 1
r = 0.05
sig = 0.6


# In[26]:


#################Q2#################
dim = 50
T = np.linspace(0, 0.8, 5)
S = np.linspace(0.8, 1.2, dim)
for t in T:
    C = [getCall(s, t, K, r, sig) for s in S]
    plt.plot(S, C, label='t = {}'.format(np.round(t, 2)))
plt.xlabel('Initial Stock Price')
plt.ylabel('Call Option Price')
plt.title('Initial Stock Price vs Call Option Price')
plt.legend()
plt.show()


# In[27]:


dim = 50
T = np.linspace(0, 0.8, 5)
S = np.linspace(0.8, 1.2, dim)
for t in T:
    P = [getPut(s, t, K, r, sig) for s in S]
    plt.plot(S, P, label='t = {}'.format(np.round(t, 2)))
plt.xlabel('Initial Stock Price')
plt.ylabel('Put Option Price')
plt.title('Initial Stock Price vs Put Option Price')
plt.legend()
plt.show()


# In[28]:


#################Q3#################
dim = 25
fig = plt.figure()
ax = fig.gca(projection='3d')

t = np.linspace(0, 0.8, dim)
s = np.linspace(0.8, 1.2, dim)

t, s = np.meshgrid(t, s)

C = np.zeros_like(t)
P = np.zeros_like(t)

for i in range(dim):
    for j in range(dim):
        C[i][j] = getCall(s[i][j], t[i][j], K, r, sig)
        P[i][j] = getPut(s[i][j], t[i][j], K, r, sig)

surfC = ax.plot_surface(t, s, C, color='b', label='Call Option')
surfC._facecolors2d=surfC._facecolors3d
surfC._edgecolors2d=surfC._edgecolors3d
surfP = ax.plot_surface(t, s, P, color='r', label='Put Option')
surfP._facecolors2d=surfP._facecolors3d
surfP._edgecolors2d=surfP._edgecolors3d

ax.legend()
ax.set_xlabel('Time')
ax.set_ylabel('Initial Stock Price')
ax.set_zlabel('Option Price')
plt.title('Option Price vs Time and Initial Stock Price')
plt.show()


# In[33]:


#################Q4#################

# ------------Varying K--------------
# =============2D Plot===============
dim = 50
T = 0.5
r = 0.05
sig = 0.6
S = np.linspace(0.8, 1.2, 5)
K = np.linspace(0.8, 1.2, dim)
table = []
plt.subplot(2, 1, 1)
for s in S:
    C = [getCall(s, T, k, r, sig) for k in K]
    plt.plot(K, C, label='S = {}'.format(s))

plt.xlabel('Strike Price')
plt.ylabel('Call Option Price')
plt.title('Strike Price vs Call Option Price')
plt.legend()
# plt.show()
plt.subplot(2, 1, 2)
for s in S:
    P = [getPut(s, T, k, r, sig) for k in K]
    plt.plot(K, P, label='S = {}'.format(s))

plt.xlabel('Strike Price')
plt.ylabel('Put Option Price')
plt.title('Strike Price vs Call Option Price')
plt.legend()
plt.show()

#-------------Tabulation--------------
S = 0.8
K = np.linspace(0.8, 1.2, 5)
C = [getCall(S, T, k, r, sig) for k in K]
P = [getPut(S, T, k, r, sig) for k in K]
table = []
for i in range(len(K)):
    table.append([K[i], C[i], P[i]])
print(tabulate(table, headers=["Strike Price", "Call Option Price", "Put Option Price"]))

# =============3D Plot===============
dim = 25
T = 0.5
r = 0.05
sig = 0.6
K = np.linspace(0.8, 1.2, dim)
S = np.linspace(0.8, 1.2, dim)

fig = plt.figure()
ax = fig.gca(projection='3d')

K, S = np.meshgrid(K, S)

C = np.zeros_like(K)
P = np.zeros_like(K)

for i in range(dim):
    for j in range(dim):
        C[i][j] = getCall(S[i][j], T, K[i][j], r, sig)
        P[i][j] = getPut(S[i][j], T, K[i][j], r, sig)

surfC = ax.plot_surface(K, S, C, color='b', label='Call Option')
surfC._facecolors2d=surfC._facecolors3d
surfC._edgecolors2d=surfC._edgecolors3d
surfP = ax.plot_surface(K, S, P, color='r', label='Put Option')
surfP._facecolors2d=surfP._facecolors3d
surfP._edgecolors2d=surfP._edgecolors3d

ax.legend()
ax.set_xlabel('Strike Price')
ax.set_ylabel('Initial Stock Price')
ax.set_zlabel('Option Price')
plt.title('Option Price vs Strike Price and Initial Stock Price')
plt.show()


# In[34]:


# ------------Varying r--------------
# =============2D Plot===============
dim = 50
T = 0.5
r = 0.05
K = 1
sig = 0.6
S = np.linspace(0.8, 1.2, 5)
R = np.linspace(0.01, 0.1, dim)
plt.subplot(2, 1, 1)
for s in S:
    C = [getCall(s, T, K, r, sig) for r in R]
    plt.plot(R, C, label='S = {}'.format(s))
plt.xlabel('Rate')
plt.ylabel('Call Option Price')
plt.title('Rate vs Call Option Price')
plt.legend()
# plt.show()

plt.subplot(2, 1, 2)
for s in S:
    P = [getPut(s, T, K, r, sig) for r in R]
    plt.plot(R, P, label='S = {}'.format(s))
plt.xlabel('Rate')
plt.ylabel('Put Option Price')
plt.title('Rate vs Put Option Price')
plt.legend()
plt.show()

#-------------Tabulation--------------
K = 1
S = 0.8
R = np.linspace(0.01, 0.1, 5)
C = [getCall(S, T, K, r, sig) for r in R]
P = [getPut(S, T, K, r, sig) for r in R]
table = []
for i in range(len(R)):
    table.append([R[i], C[i], P[i]])
print(tabulate(table, headers=["Rate", "Call Option Price", "Put Option Price"]))

# =============3D Plot===============
dim = 25
T = 0.5
K = 1
sig = 0.6
r = np.linspace(0.01, 0.1, dim)
S = np.linspace(0.8, 1.2, dim)

fig = plt.figure()
ax = fig.gca(projection='3d')

r, S = np.meshgrid(r, S)

C = np.zeros_like(r)
P = np.zeros_like(r)

for i in range(dim):
    for j in range(dim):
        C[i][j] = getCall(S[i][j], T, K, r[i][j], sig)
        P[i][j] = getPut(S[i][j], T, K, r[i][j], sig)

surfC = ax.plot_surface(r, S, C, color='b', label='Call Option')
surfC._facecolors2d=surfC._facecolors3d
surfC._edgecolors2d=surfC._edgecolors3d
surfP = ax.plot_surface(r, S, P, color='r', label='Put Option')
surfP._facecolors2d=surfP._facecolors3d
surfP._edgecolors2d=surfP._edgecolors3d

ax.legend()
ax.set_xlabel('Rate')
ax.set_ylabel('Initial Stock Price')
ax.set_zlabel('Option Price')
plt.title('Option Price vs Rate and Initial Stock Price')
plt.show()


# In[36]:


# ------------Varying sig--------------
# ==============2D Plot================
dim = 50
T = 0.5
r = 0.05
K = 1
S = np.linspace(0.8, 1.2, 5)
Sig = np.linspace(0.1, 1.0, dim)

plt.subplot(2, 1, 1)
for s in S:
    C = [getCall(s, T, K, r, sig) for sig in Sig]
    plt.plot(Sig, C, label='S = {}'.format(s))
plt.xlabel('Standard Deviation')
plt.ylabel('Call Option Price')
plt.title('Standard Deviation vs Call Option Price')
plt.legend()
# plt.show()

plt.subplot(2, 1, 2)
for s in S:
    P = [getPut(s, T, K, r, sig) for sig in Sig]
    plt.plot(Sig, P, label='S = {}'.format(s))
plt.xlabel('Standard Deviation')
plt.ylabel('Call Option Price')
plt.title('Standard Deviation vs Put Option Price')
plt.legend()
plt.show()

#-------------Tabulation--------------
S = 0.8
r = 0.05
Sig = np.linspace(0.1, 1.0, 5)
C = [getCall(S, T, K, r, sig) for sig in Sig]
P = [getPut(S, T, K, r, sig) for sig in Sig]
table = []
for i in range(len(Sig)):
    table.append([Sig[i], C[i], P[i]])
print(tabulate(table, headers=["Standard Deviation", "Call Option Price", "Put Option Price"]))

# =============3D Plot===============
dim = 25
T = 0.5
K = 1
r = 0.05
Sig = np.linspace(0.1, 1.0, dim)
S = np.linspace(0.8, 1.2, dim)

fig = plt.figure()
ax = fig.gca(projection='3d')

Sig, S = np.meshgrid(Sig, S)

C = np.zeros_like(Sig)
P = np.zeros_like(Sig)

for i in range(dim):
    for j in range(dim):
        C[i][j] = getCall(S[i][j], T, K, r, Sig[i][j])
        P[i][j] = getPut(S[i][j], T, K, r, Sig[i][j])

surfC = ax.plot_surface(Sig, S, C, color='b', label='Call Option')
surfC._facecolors2d=surfC._facecolors3d
surfC._edgecolors2d=surfC._edgecolors3d
surfP = ax.plot_surface(Sig, S, P, color='r', label='Put Option')
surfP._facecolors2d=surfP._facecolors3d
surfP._edgecolors2d=surfP._edgecolors3d

ax.legend()
ax.set_xlabel('Standard Deviation')
ax.set_ylabel('Initial Stock Price')
ax.set_zlabel('Option Price')
plt.title('Option Price vs Standard Deviation and Initial Stock Price')
plt.show()


# In[ ]:




