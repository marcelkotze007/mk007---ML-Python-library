import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The data (X1, X2, X3) are for each patient
# X1 = The systolic blood pressure (what we want to predict)
# X2 = age in years
# X3 = weight in pounds
class systolic:
    def get_data(self):
        dataset = pd.read_excel("C:/Users/Marcel/OneDrive/Python Courses/Deep Learning/Linear Regression/Data/Systolic.xls")
        X = dataset.values

        dataset['ones'] = 1
        dataset = dataset
        Y = dataset['X1']
        X = dataset[['X2', 'X3', 'ones']]

        X2only = dataset[['X2', 'ones']]
        X3only = dataset[['X3', 'ones']]
        return X, Y, X2only, X3only

    def calc_weights(self, X, Y):
        """
        Calculates the weights of the model. 
        Returns w and Yhat
        """
        w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y)) #Use np.dot() to do matrix multiplication
        Yhat = np.dot(X, w)  #X must be first so the inner dimensions are the same
        
        return Yhat

    def R_squared(self, Y, Yhat):
        """
        Calculates the R squared that indicates how well the model fits the data.
        The closer R squared is to 1, the better the fit. 
        Must first calculate the expected values of Y i.e. Yhat, using calculate_a_b() function
        """
        dif1 = Y - Yhat
        dif2 = Y - Y.mean()
        
        #Rater use the dot function as it multiplies and sums the values, vector multiplication,
        #as 100x1 * 1x100 = a single value, the sum                                    
        R = 1 - dif1.dot(dif2)/dif2.dot(dif2) 
        print("The value of R squared is", R)
        return R

Sys = systolic()

X, Y, X2only, X3only = Sys.get_data()
Yhat = Sys.calc_weights(X, Y)
Sys.R_squared(Y, Yhat)

