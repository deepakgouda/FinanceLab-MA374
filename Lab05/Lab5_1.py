import matplotlib.pyplot as plt
import numpy as np

mu = np.array([0.1, 0.2, 0.15])
cov = np.array([[0.005, -0.010, 0.004], 
				[-0.010, 0.040, -0.002], 
				[0.004, -0.002, 0.023]])
rf = 0.1
dim = len(mu)
u = np.ones((1, dim))

n = 500

def efficientFrontier(Y, color="red", 
				label="Efficient Frontier"):
	W = np.zeros((len(Y), 3))
	X = []
	i = 0
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
		
		W[i][0] = w[0][0]
		W[i][1] = w[0][1]
		W[i][2] = w[0][2]
		X.append(sig_v)
		i = i+1
	# plt.subplot(2, 1, 1)
	plt.scatter(X, Y, marker=".", color=color, label=label)
	return W

def model(mu, cov, color, label):
	X = np.zeros(n)
	Y = np.zeros(n)
	W = np.zeros((n, 2))
	length = len(mu)

	np.random.seed(0)
	for i in range(n):
		w = np.random.uniform(0, 1, (1, length))
		w = w/np.sum(w)
		
		W[i][0] = w[0][0]
		W[i][1] = w[0][1]
		
		X[i] = np.matmul(np.matmul(w, cov)[0], np.transpose(w))
		X[i] = X[i]**0.5
		Y[i] = np.matmul(mu, np.transpose(w))
	# plt.subplot(2, 1, 1)
	plt.scatter(X, Y, marker='.', color=color, label=label)
	return W

mu1_2 = mu[:2]
cov1_2 = cov[:2, :2]

mu1_3 = [mu[i] for i in range(len(mu)) if i!= 1]
cov1_3 = np.array([[cov[0][0], cov[0][2]], 
					[cov[2][0], cov[2, 2]]])

mu2_3 = mu[1:]
cov2_3 = cov[1:, 1:]

efficientMu = np.linspace(0.08, 0.20, n*10)

W = model(mu, cov, "blue", "Stock 1, 2 & 3")
model(mu1_2, cov1_2, "magenta", "Stock 1, 2")
model(mu1_3, cov1_3, "yellow", "Stock 1, 3")
model(mu2_3, cov2_3, "cyan", "Stock 2, 3")

W_ef = efficientFrontier(efficientMu)
plt.legend()
plt.title("Minimum Variance Curve and Efficient Frontier")
plt.xlabel("Volatility")
plt.xlabel("Return")
plt.show()

# print(W)
# print(np.shape(W))
X_ef = W_ef[:, 0]
Y_ef = W_ef[:, 1]

for i in range(len(X_ef)):
	if X_ef[i] >=0 and Y_ef[i] >=0:
		plt.scatter(X_ef[i], Y_ef[i], color="red")

X = W[:, 0]
Y = W[:, 1]
plt.scatter(X, Y, color="blue", marker='.')

plt.title("W1 vs W2")
plt.xlabel("W1")
plt.ylabel("W2")
plt.show()
