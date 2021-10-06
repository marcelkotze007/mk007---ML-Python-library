import numpy as np
import matplotlib.pyplot as plt

N = 10
D = 3  #Dimensionality
X = np.zeros((N,D))

X[:, 0] = 1   #bais column
X[:5, 1] = 1  #Sets the first five rows of column 2 to 1
X[5:, 2] = 1  #Sets the last five rows of column 3 to 1

#print(X)

Y = np.array([0] * 5 + [1] * 5)  #Sets Y = 0, for first half of the data and Y = 1 for the second half of the data
#print(Y)

costs = []    #Store the costs in able to draw them later
w = np.random.randn(D)/np.square(D) #Creates the random weights ensuring that it has variance 1/D

learning_rate = 0.001

for t in range(1000):  #Number of iterations
    Y_hat = X.dot(w)
    delta = Y_hat - Y
    w = w - learning_rate * X.T.dot(delta)
    mse = delta.dot(delta)/N
    costs.append(mse)

plt.plot(costs)
plt.show()

print(w)

plt.plot(Y_hat, label = 'predictions')
plt.plot(Y, label = "targets")
plt.legend()
plt.show()
