import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import numpy as np
from tabulate import tabulate
def getStock(stockName, market):
	# if market is 'BSE':
		# fields = ['Close Price', 'Open Price']
	# else:
	fields = ['Close', 'Open']
	dataframe = read_csv('./'+market+'Data/'+stockName+'.csv', 
							skipinitialspace = True, 
							squeeze = True, 
							usecols = fields, 
							index_col=False)
	dataframe = (dataframe[fields[1]] - 
				dataframe[fields[0]])/dataframe[fields[0]]
	return dataframe

def getIndex(market):
	fields = ['Close', 'Open']
	dataframe = read_csv(market.lower()+"data1.csv", 
				skipinitialspace = True, 
				squeeze = True, 
				usecols = fields, 
				index_col=False)
	dataframe = (dataframe[fields[1]] - 
				dataframe[fields[0]])/dataframe[fields[0]]
	return dataframe

def getBeta(stock, index):
	var = np.var(index)
	covar = 0
	n = len(stock)
	stock_mean = np.mean(stock)
	index_mean = np.mean(index)
	for i in range(n):
		covar = covar + (stock[i]-stock_mean)*(index[i]-index_mean)
	covar = covar/(n-1)

	# print(covar/var)
	# print((np.cov(stock, index)[0, 1])/(np.cov(stock, index)[1, 1]))
	# print(var)
	return covar/var

def getAnnualReturns(data):
	daily_ret = []
	annual_ret = []
	daily_ret.append(data[:244])
	daily_ret.append(data[244 : 491])
	daily_ret.append(data[492 : 738])
	daily_ret.append(data[739 : 986])
	daily_ret.append(data[987 : 1232])

	daily_ret = np.array(daily_ret)

	for i in range(5):
		annual_ret.append(np.sum(daily_ret[i]))

	annual_ret = np.array(annual_ret)
	return annual_ret

def plot(market, stock_name):
	dataframe = getIndex(market)
	ret = np.array(dataframe)

	# daily_ret = []
	# annual_ret = []
	# daily_ret.append(ret[:244])
	# daily_ret.append(ret[244 : 491])
	# daily_ret.append(ret[492 : 738])
	# daily_ret.append(ret[739 : 986])
	# daily_ret.append(ret[987 : 1232])

	# daily_ret = np.array(daily_ret)

	# for i in range(5):
	# 	annual_ret.append(np.sum(daily_ret[i]))

	annual_ret = getAnnualReturns(ret)

	print("Mean = "+str(np.round(np.mean(annual_ret), 3)))
	print("Var = "+str(np.round(np.var(annual_ret), 3)))
	
	dim = len(stock_name)
	rf = 0.05
	u = np.ones((1, dim))

	stocks=[]
	mu = []
	
	for name in stock_name:
		dataframe = getStock(name, market)
		data = np.array(dataframe)
		data = getAnnualReturns(data)
		stocks.append(data)
		mu.append(np.mean(data))

	stocks = np.array(stocks)
	mu = np.array(mu)

	# print("Company\t\tBeta Value\t\tActual Return\t\tExpected Return")
	table = []
	for i in range(len(stock_name)):
		stock = stocks[i]
		index = annual_ret
		beta_v = np.round(getBeta(stock, index), 3)
		mu_v = np.round((rf + (mu[i]-rf)*beta_v), 3)
		table.append([stock_name[i], beta_v, np.round(mu[i], 3), mu_v])

	# for i in range(len(X)):
	# 	table.append([X[i], Y[i]])
	print(tabulate(table, headers=["Company Name", "beta_v", "Actual Return", "Expected Return"]))
		# print(stock_name[i]+'\t\t'+str(beta_v)+'\t\t'+str(np.round(mu[i], 3))+'\t\t'+str(mu_v))

# stock_name=['ABB']
stock_name=['ABB', 'AXISBANK', 'BALMLAWRIE', 'BHARTIARTL', 
			'CUMMINSIND', 'EMAMILTD', 'GODREJIND', 
			'HDFCBANK', 'HEROMOTOCO', 'HINDUNILVR', 
			'INFY', 'IOC', 'ITC', 'LUPIN', 'M&M', 
			'MAHABANK', 'NTPC', 'SBIN', 'SHREECEM', 'TCS']

print("\t\t BSE Data")
plot('BSE', stock_name)
print()
print("\t\t NSE Data")
plot('NSE', stock_name)
