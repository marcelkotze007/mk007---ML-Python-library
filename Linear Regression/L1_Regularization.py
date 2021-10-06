import numpy as np
import matplotlib.pyplot as plt

N = 50
D = 50  #not ideal situation ->  fat matrix X = N x D

"""
Little demonstration of how the data is distributed
N = 5
D = 5
X = np.random.random((N,D))
Y = (np.random.random((N,D)) - 0.5)
Z = (np.random.random((N,D)) - 0.5) * 10
print(X, Y, Z)
"""
"""
#Little demonstration of the function of sign in the gradient decent:
#THere to add a bais term to the function in the form of -1, 0, 1
X = np.random.randn(10)
print(np.sign(X))
"""
#Creates a uniform ditribution of size N x D, centered around 0, from -5 to +5
X = (np.random.random((N,D)) - 0.5) * 10 
#print(X)

true_w = np.array([1, 0.5, -.5] + [0]*(D-3))
#print(true_w)

#Plus guassian distribution as noise, since it is part of linear regression 
Y = X.dot(true_w) + np.random.randn(N)*0.5 

#Next step is to do Gradiant decent:
costs = []
w = np.random.randn(D) / np.sqrt(D)
learning_rate = 0.001
L1 = 10
#Start the gradient decent by running through the itterations
for t in range(1000):
    Y_hat = np.dot(X, w)
    delta = Y_hat - Y
    #Have to use the transpose of X so: D x N * N * 1
    #Thus, given w = D x 1
    w = w - learning_rate * (X.T.dot(delta) + L1*np.sign(w))
    
    #Next step is to find and store the cost of the function, incdicated by mse (mean squared error) or J
    # mse = (Y_hat - Y)^2 / N
    mse = delta.dot(delta) / N
    costs.append(mse)

plt.plot(costs)
plt.show()

#print("Final w: "w)

plt.plot(true_w, label = "true w")
plt.plot(w, label = 'w_map')
plt.legend()
plt.show()
