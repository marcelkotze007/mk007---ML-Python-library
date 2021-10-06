import numpy as np
import matplotlib.pyplot as plt

#Create the data
class L2_Regularization():
    def get_data(self):
        N = 50
        X = np.linspace(0, 10, N)
        Y = 0.5*X + np.random.randn(N)
        
        #Creating outlyers to the data
        Y[-1] += 30
        Y[-2] += 30

        return X, Y

    def solve_w(self, X, Y, N = 50):
        X = np.vstack([np.ones(N), X]).T

        w_ml = np.linalg.solve(X.T.dot(X), X.T.dot(Y))

        Y_hat_ml = X.dot(w_ml)

        return Y_hat_ml, X
    
    def L2(self, X):
        L2 = 1000    #The penalty value
        w_map = np.linalg.solve(L2*np.eye(2) + X.T.dot(X), X.T.dot(Y))  #Solves w and adds the weight penalty
        Y_hat_map = X.dot(w_map)

        return Y_hat_map
    
L2_Reg = L2_Regularization()

X, Y = L2_Reg.get_data()
plt.scatter(X, Y)
plt.show()

Y_hat_ml, X = L2_Reg.solve_w(X, Y)
plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Y_hat_ml)
plt.show()

Y_hat_map = L2_Reg.L2(X)
plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Y_hat_map, label = "L2 - MAP")
plt.plot(X[:, 1], Y_hat_ml, label = "max likelihood")
plt.legend()
plt.show()