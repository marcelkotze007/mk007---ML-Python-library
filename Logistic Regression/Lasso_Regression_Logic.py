import numpy as np
import matplotlib.pyplot as plt

#The purpose of L1 Regularization is to reduce the number of input(features) by given them a weight of 0

class lasso_regression:
    def create_data(self, N = 50, D = 50):
        X = (np.random.random((N, D)) - 0.5) * 10

        #X[:25,:] = X[:25,:] + 5 * np.ones((N, D))
        #X[25:,:] = X[25:,:] - 5 * np.ones((N,D))

        #Setting up the weights that matter
        true_w = np.array([1, 0.5, -0.5] + [0] * (D - 3))

        Y = np.round(LR.sigmoid(X.dot(true_w) + np.random.randn(N)*0.5))

        return X, Y, true_w, D

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy_error(self, Y_hat, Y, w, l1):
        return -np.mean(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat)) + l1*np.mean(np.abs(w))

    def gradient_decent(self, learning_rate = 0.01, l1 = 2):
        costs = []

        X, Y, true_w, D = LR.create_data()

        w = np.random.randn(D)/np.sqrt(D)

        #setting up the penalty, encouraged to try different values:
        #l1 = 2.0 change it in the function

        for i in range(5000):
            Y_hat = LR.sigmoid(X.dot(w))
            delta = Y_hat - Y
            cost_train = LR.cross_entropy_error(Y_hat, Y, w, l1)

            w -= learning_rate * (X.T.dot(delta) + l1 * np.sign(w))

            costs.append(cost_train)
            if i % 500 == 0:
                print(i, cost_train)
        
        print("final w:", w)

        plt.plot(costs)
        plt.show()

        plt.plot(true_w, label = 'true w')
        plt.plot(w, label = 'w MAP')
        plt.legend()
        plt.show()

LR = lasso_regression()

l1_value = float(input("Enter l1 value: "))
learn_value = float(input("Enter learning rate (recommended: 0.001): "))


LR.gradient_decent(learning_rate=learn_value, l1=l1_value)