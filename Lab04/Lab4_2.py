import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import numpy as np

stock_name = ['AAPL', 'ADS', 'AMZN', 'DE', 'FB', 
			'GOOGL', 'IBM', 'MSFT', 'NKE', 'NVDA']

dim = len(stock_name)
length = 50

mask = np.linspace(0, 1000, 50)
mask = np.round(mask, 0)

stock=[]
mu = []


fields = ['close', 'open']
for name in stock_name:
	dataframe = read_csv('./Data/'+name+'_data.csv', 
				skipinitialspace = True, squeeze = True, 
				usecols = fields)
	dataframe = (dataframe['close'] - dataframe['open'])
	data = [dataframe[i] for i in mask]
	data = np.array(data)
	stock.append(data)
	mu.append(np.mean(data))

stock = np.array(stock)
mu = np.array(mu)
cov = np.cov(stock)

stock_df = pd.DataFrame(np.transpose(np.round(stock, 3)))
stock_df.to_csv('Data.csv', index=False, header=stock_name)

rf = 0.07
u = np.ones((1, dim))

def model(Y):
	W = []
	X = []
	for mu_v in Y:

		ucimt = np.matmul(u, np.linalg.inv(cov))
		ucimt = np.matmul(ucimt, np.transpose(mu))

		uciut = np.matmul(u, np.linalg.inv(cov))
		uciut = np.matmul(uciut, np.transpose(u))

		mcimt = np.matmul(mu, np.linalg.inv(cov))
		mcimt = np.matmul(mcimt, np.transpose(mu))

		mciut = np.matmul(mu, np.linalg.inv(cov))
		mciut = np.matmul(mciut, np.transpose(u))
		
		mat_1 = np.array([[1, ucimt], [mu_v, mcimt]])
		mat_2 = np.array([[uciut[0, 0], 1], [mciut, mu_v]])
		mat_3 = np.array([[uciut[0, 0], ucimt], [mciut, mcimt]])
		
		det_1 = np.linalg.det(mat_1)
		det_2 = np.linalg.det(mat_2)
		det_3 = np.linalg.det(mat_3)

		w = det_1*(np.matmul(u, 
			np.linalg.inv(cov)))
		w = w + det_2*(np.matmul(mu, 
			np.linalg.inv(cov)))
		w = w/det_3
		
		sig_v = np.matmul(np.matmul(w, cov), 
							np.transpose(w))
		sig_v = sig_v**0.5
		
		W.append(w)
		X.append(sig_v)
	return W, X

def marketPortfolio():
	w = np.matmul(mu - rf*u, np.linalg.inv(cov))
	w = w/np.sum(w)
	return w

iter = 1000
ymax = 2
ymin = -2
Y = np.linspace(ymin, ymax, iter)
W, X = model(Y)

plt.scatter(X, Y, marker=".", color="orange")

idealW = marketPortfolio()
idealMu = np.dot(idealW, mu)
idealSig = np.matmul(np.matmul(idealW, cov), 
						np.transpose(idealW))
idealSig = idealSig**0.5

plt.scatter(idealSig, idealMu, color="blue")
plt.scatter(0, rf, color="blue")
xmin = (idealSig+idealSig*(ymin-idealMu)/(idealMu-rf))[0]
plt.plot(np.array([xmin, 0]), 
		np.array([ymin, rf]), color='blue')
plt.annotate("Market Portfolio("+str(idealSig)+","+
					str(idealMu)+")", (idealSig, idealMu))
plt.annotate("Zero Risk Portfolio("+str(0)+","+
					str(rf)+")", (0, rf))
plt.title("Markowitz Efficient Frontier Line and CAPM Line")
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.show()

for i in range(len(mu)):
	X = np.linspace(-1, 1, 100)
	Y = rf + (i - rf)*X
	plt.plot(X, Y, label=stock_name[i])
plt.title("Security Market Lines")
plt.xlabel("Beta")
plt.ylabel("Mean Return")
plt.legend()
plt.show()
