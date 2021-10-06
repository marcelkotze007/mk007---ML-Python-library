import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Multiple_Linear_Regression import Multiple_Linear

Multi_Lin = Multiple_Linear()

class polynomial_linear:
    def get_data(self, filename):
        """
        Loads the data.
        Takes the filename as an argument.
        Returns all of the input in array X
        Returns the output as an array Y
        """
        dataset = pd.read_csv(filename).values

        #Checks how many interations there are and assigns value N
        N = np.array(np.shape(dataset))
        N = N[0] 

        X = dataset[:, 0]
        X_squared = X*X
        X = np.c_[np.ones(N), X, X_squared]
        Y = dataset[:, 1]

        return X, Y

filename = "C:/Users/Marcel/OneDrive/Python Courses/Deep Learning/Linear Regression/Data/data_poly.csv"

Poly_Lin = polynomial_linear()

X, Y = Poly_Lin.get_data(filename)
Yhat = Multi_Lin.calc_weights(X, Y)   #Calculates the weights using the function we created in Multi_Lin
Multi_Lin.R_squared(Y, Yhat)

plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Yhat))  #Need to sort for X in order to avoid drawing lines to each value
plt.show()