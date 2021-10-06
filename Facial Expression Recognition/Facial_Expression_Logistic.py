import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.utils import shuffle
from Utilities import functions

Fun = functions()

filename = "D:/Machine Learning datasets/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013.csv"
#filename_test = "D:/Machine Learning datasets/challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013_test.csv"
#When using the filename_test, X_valid and Y_valid should be set to X[-250:], Y[-250:]

class logistic_model(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y, learning_rate = 10e-7, reg = 0, epochs = 120000, show_fig = False):
        """
        Recommended epochs is 120 000, with learning_rate = 10e-7
        Does the gradient decent and plots the Cross-Entropy error cost if show_fig = True
        """
        dt0 = dt.now()
        X, Y = shuffle(X, Y)
        X_valid, Y_valid = X[-1000:], Y[-1000:]

        D = X.shape[1]
        self.W = np.random.randn(D)/np.sqrt(D)
        self.b = 0

        costs = []
        best_validation_error = 1

        for i in range(epochs):
            Y_hat = Fun.sigmoid(X.dot(self.W) + self.b)

            #Gradient decent:
            self.W += learning_rate * (X.T.dot(Y - Y_hat) - reg * self.W)
            self.b += learning_rate * ((Y - Y_hat).sum() - reg * self.b)

            Y_hat_valid = Fun.sigmoid(X_valid.dot(self.W) + self.b)
            cost_valid = Fun.cross_entropy_error(Y_valid, Y_hat_valid)
            costs.append(cost_valid)
            e = Fun.error_rate(Y_valid, Y_hat_valid)
            if e < best_validation_error:
                best_validation_error = e

            if i % 1000 == 0:
                print("i: %s cost: %s error: %s" %(i, cost_valid, e))
        
        class_rate_valid = Fun.classification_rate(Y_valid, np.round(Y_hat_valid))

        print("Final classification rate: ", class_rate_valid)
        print("Best validation error: ", best_validation_error)
        dt1 = dt.now()
        time = dt1 - dt0
        print("time: ", time)

        if show_fig:
            plt.plot(costs, label = "Cross-Entropy Error Cost")
            plt.legend()
            plt.show()
        
    def score(self, X, Y, show = False):
        prediction = np.round(Fun.sigmoid(X.dot(self.W) + self.b))
        score = 1 - Fun.error_rate(Y, prediction) 
        if show:
            print(score)
        return score

def main():
    X, Y = Fun.get_binary_data()

    #Adjust for classification dispersion
    X0 = X[Y == 0, :]
    X1 = X[Y == 1, :]
    #We know there are 9 times less of class 1 than class 0
    X1 = np.repeat(X1, 9, axis= 0)
    X = np.vstack([X0, X1])
    Y = np.array([0] * len(X0) + [1] * len(X1))

    model = logistic_model()
    model.fit(X, Y, show_fig=True)
    model.score(X, Y, show = True)

if __name__ == "__main__":
    main()