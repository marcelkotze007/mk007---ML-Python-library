import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D #Used to sketch 3D graphs
import matplotlib.pyplot as plt

class Multiple_Linear(object):
    def get_data(self, filename):
        """
        Loads the data.
        Takes the filename as an argument.
        Returns all of the input in array X
        Returns the output as an array Y
        """
        dataset = pd.read_csv(filename).values
        #print(np.shape(dataset))
        N = np.array(np.shape(dataset)) #Converts the shape into a np array
        #print(np.shape(N))             #Used to check the shape of the array
        N = N[0]                        #Gets the number of rows
        #print(N)
        #X1 = dataset[:, 0]
        X2 = dataset[:, :-1] #The code to add more than one column and leave out the last column (usualy the Y values)
        X = np.c_[X2, np.ones(N)] #The code to add an additional column to a np array
        Y = dataset[:, -1]   #Selects the last column of the data
        #print(Y)

        return X, Y

    def get_data_for_loop(self):
        X = []
        Y = []
        for line in open("C:/Users/Marcel/OneDrive/Python Courses/Deep Learning/Linear Regression/Data/data_2d.csv"):
            x1, x2, y = line.split(',')
            X.append([float(x1), float(x2), 1])
            Y.append(float(y))
        
        X = np.array(X)
        Y = np.array(Y)

        return X, Y 

    def plot_graph(self, X, Y):
        """
        Used to plot 3D scatter plots
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(X[:, 0], X[:, 1], Y)
        plt.show()

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
        

filename = "C:/Users/Marcel/OneDrive/Python Courses/Deep Learning/Linear Regression/Data/data_2d.csv"

MultiL = Multiple_Linear()
X, Y = MultiL.get_data(filename)
#X, Y = MultiL.get_data_for_loop()
MultiL.plot_graph(X, Y)
Yhat = MultiL.calc_weights(X, Y)
MultiL.R_squared(Y, Yhat)
