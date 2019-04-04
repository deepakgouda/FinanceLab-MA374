import matplotlib.pyplot as plt
import numpy as np

mu = np.array([0.1, 0.2, 0.15])
cov = np.array([[0.005, -0.010, 0.004], 
				[-0.010, 0.040, -0.002], 
				[0.004, -0.002, 0.023]])
rf = 0.1
dim = len(mu)
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
Y = np.linspace(0.005, 0.3, iter)
W, X = model(Y)

print("Portfolio(without riskfree assets) at 15% risk")
print("Index\tWeight\t\t\t\t\tReturn\t\t\tRisk")
tol = 0.0003
for i in range(len(X)):
	if abs(X[i]-0.15) < tol:
		print(i, W[i], Y[i], X[i])
print()

print("Portfolio(without riskfree assets) at 18% return")
print("Index\tWeight\t\t\t\t\tReturn\t\t\tRisk")
tol = 0.00015
for i in range(len(X)):
	if abs(Y[i]-0.18) < tol:
		print(i, W[i], Y[i], X[i])
print()

plt.scatter(X, Y, marker=".", color="orange")
plt.axvline(x=0.15, color="red")
plt.axhline(y=0.18, color="blue")
plt.text(0.15,-0.01,'x = 0.15')
plt.text(-0.02,0.18,'y = 0.18')

indx = np.linspace(0, len(W)-1, 10)
indx = np.round(indx, 0)
indx = [int(i) for i in indx]

print("Index\tWeight\t\t\t\t\tReturn\t\t\tRisk")
for i in indx:
	print(str(i)+"\t"+str(W[i])+"\t"+str(Y[i])+"\t"+str(X[i]))
print()

idealW = marketPortfolio()
idealMu = np.dot(idealW, mu)
idealSig = np.matmul(np.matmul(idealW, cov), 
						np.transpose(idealW))
idealSig = idealSig**0.5

risk1 = 0.10
c1 = risk1/idealSig
w1 = np.append(c1*idealW, (1-c1)*1)
print("Portfolio(with risky and riskfree assets) at "+str(risk1)+"0% risk = ", end='')
print(w1)

risk2 = 0.25
c2 = risk2/idealSig
w2 = np.append(c2*idealW, (1-c2)*1)
print("Portfolio(with risky and riskfree assets) at "+str(risk2)+"% risk = ", end='')
print(w2)

plt.scatter(idealSig, idealMu, color="blue")
plt.scatter(0, rf, color="blue")

ymax = 0.3
xmax = idealSig+idealSig*(ymax-idealMu)/(idealMu-rf)


plt.plot(np.array([0, xmax]), 
		np.array([rf, ymax]), color='blue')
plt.annotate("Market Portfolio("+str(idealSig)+","+
					str(idealMu)+")", (idealSig, idealMu))
plt.annotate("Zero Risk Portfolio("+str(0)+","+
					str(rf)+")", (0, rf))
plt.title("Markowitz Efficient Frontier Line and CAPM Line")
plt.xlabel("Volatility")
plt.ylabel("Return")
plt.show()
