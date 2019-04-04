import numpy as np
import matplotlib.pyplot as plt

X = range(5, 51, 5)
Y = [9.06572, 10.0341, 10.491, 10.7788, 10.2972, 10.4849, 10.4828, 11.2478, 10.2849, 10.5368]

for i in X:
	print(i)

plt.title("Markov Method : Call Option Price vs M")
plt.xlabel("M")
plt.ylabel("Option Price")
plt.plot(X, Y, '-o')
plt.show()