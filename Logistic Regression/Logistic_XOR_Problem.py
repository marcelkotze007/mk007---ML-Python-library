import numpy as np
import matplotlib.pyplot as plt

#The key to solving this problem is to turn it into a 3D-problem ans draw a plan between the 2 datasets

class XOR:
    def create_data(self, N = 4, D = 2):
        
        #XOR output for 2 inputs and 1 output
        #Input = 1 1, output = 0
        #Input = 0 0, output = 0
        #Input = 0 1, output = 1
        #Input = 1 0, output = 1
        #Input:
        X = np.array([
            [0, 0],
            [0, 1],
            [1, 0],
            [1, 1],
        ])
        #Output:
        T = np.array([0, 1, 1, 0])

        ones = np.ones((N, 1))

        #Showcases the issue with using logistic regression, as only 50% data can be separated with a line at a time
        #plt.scatter(X[:, 0], X[:,1], c = T)
        #plt.show()

        #xy multiplies the X with the T (Y values) to give a new range
        xy = (X[:,0] * X[:,1]).reshape(N,1)

        X_bias = np.concatenate((ones, xy, X), axis = 1)

        return X_bias, T

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cross_entropy_error(self, Y, Y_hat):
        return -np.mean(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

    def gradient_decent(self, learn_rate = 0.01, l2 = 0.01):
        error = []
        X_bias, T = XOR.create_data()
        N, D = X_bias.shape
        w = np.random.randn(D)

        for i in range(10000):
            z = X_bias.dot(w)
            Y_hat = XOR.sigmoid(z)

            e = XOR.cross_entropy_error(T, Y_hat)
            if i % 500 == 0:
                print(e)

            error.append(e)

            w += learn_rate * (X_bias.T.dot(T - Y_hat) - l2*w)

        plt.plot(error)
        plt.title("Cross-Entropy Error per iteration")
        plt.show()

        print("Final w: ",w)
        print("Final classification rate: ", 1 - np.abs(T - np.round(Y_hat)).sum()/N)

XOR = XOR()

XOR.gradient_decent()