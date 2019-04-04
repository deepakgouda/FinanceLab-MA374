import numpy as np
import matplotlib.pyplot as plt
from math import exp
from tabulate import tabulate

S0 = 100.0
K = 105.0
T = 5
r = 0.05
sig = 0.3

def model(M = [1, 5, 10, 20, 50, 100, 200, 400], 
						option = "Put", 
						plot = True, 
						priceTable = False, 
						timeTable = False):
	PriceList = []
	for N in M:
		dt = T/N
		u = exp(sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
		d = exp(-sig*(dt**0.5) + (r - 0.5*(sig**2))*dt)
		
		t = T/N
		q = (exp(-r*t) - d) / (u-d)

		dim = N+1

		Stock = np.zeros((dim, dim))
		P = np.zeros((dim, dim))
		Payoff = np.zeros((dim, dim))

		Stock[0][0] = S0
		P[0][0] = 1

		for j in range(1, dim):
			for i in range(j):
				Stock[i][j] = Stock[i][j-1]*u
				Stock[i+1][j] = Stock[i][j-1]*d
				P[i][j] = P[i][j-1]*u
				P[i+1][j] = P[i][j-1]*d
		Stock = np.round(Stock, decimals = 2)

		if option is "Put":
			for i in range(dim):
				Payoff[i][dim-1] = max(0, K - P[i][dim-1]*S0)
		else:
			for i in range(dim):
				Payoff[i][dim-1] = max(0, P[i][dim-1]*S0 - K)

		for j in range(dim-2, -1, -1):
			for i in range(j+1):
				Payoff[i][j] = exp(-r*t)*(q*Payoff[i][j+1] + 
									(1-q)*Payoff[i+1][j+1])
		PriceList.append(Payoff[0][0])

	if plot:
		X = M
		Y = PriceList
		plt.xlabel("Number of Steps")
		plt.ylabel("Price of Put Option")
		plt.title(option+" Option Pricing")
		plt.plot(X, Y)
		plt.show()

	if priceTable:
		print("\n\t"+option+" Price")
		table = []
		X = M
		Y = PriceList
		for i in range(len(X)):
			table.append([X[i], Y[i]])
		print(tabulate(table, headers=["Step Size", "Option Price"]))
	elif timeTable:
		print("\n\t"+option+" Price")
		Tabulation = np.array([0, 0.50, 1, 1.50, 3, 4.5])
		X = [int(i/dt) for i in Tabulation]
		Y = [Payoff[i] for i in X]
		table = []
		for i in range(len(X)):
			table.append([X[i], Y[i]])
		print(tabulate(table, headers=["Step Size", "Option Price"]))


M = [1, 5, 10, 20, 50, 100, 200, 400]
model(M, "Put", plot=False, priceTable=True, 
							timeTable = False)
model(M, "Call", plot=False, priceTable=True, 
							timeTable = False)

M = range(5, 100, 5)
model(M, "Put", plot=True, priceTable=False, 
							timeTable = False)

M = range(1, 100, 1)
model(M, "Put", plot=True, priceTable=False, 
							timeTable = False)

M = range(5, 100, 5)
model(M, "Call", plot=True, priceTable=False, 
							timeTable = False)

M = range(1, 100, 1)
model(M, "Call", plot=True, priceTable=False, 
							timeTable = False)

M = [20]
model(M, "Put", plot=False, priceTable=False, 
							timeTable = True)
model(M, "Call", plot=False, priceTable=False, 
							timeTable = True)