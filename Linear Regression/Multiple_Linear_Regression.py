import numpy as np
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D #Used to sketch 3D graphs
import matplotlib.pyplot as plt

class Multiple_Linear(object):
    def get_data(self, filename, test_size):
        """
        Loads the data.
        Takes the filename as an argument.
        Returns all of the input in array X
        Returns the output as an array Y
        """
        dataset = pd.read_csv(filename).values
        #print(dataset.head())
        #print(np.shape(dataset))
        N = np.array(np.shape(dataset)) #Converts the shape into a np array
        #print(np.shape(N))             #Used to check the shape of the array
        N = N[0]                        #Gets the number of rows
        D = dataset.shape[1] - 1
        #print(N)
        #X1 = dataset[:, 0]
        X2 = dataset[:, :-2] #The code to add more than one column and leave out the last column (usualy the Y values)
        X = np.c_[X2, np.ones(N)] #The code to add an additional column to a np array
        X_train = X[:test_size]
        X_test = X[test_size:]
        Y = dataset[:, -2]   #Selects the last column of the data
        Y_train = Y[:test_size]
        Y_test = Y[test_size:]
        #print(Y)

        return X_train, Y_train, X_test, Y_test, N, D

    def get_data_for_loop(self):
        X = []
        Y = []
        for line in open(r"C:\Dev\SourceCode\Personal\mk007---ML-Python-library\data\archive\hou_all.csv"):
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

    def calc_weights(self, X, Y, Ridge = False, L2=100, Feat=2):
        """
        Calculates the weights of the model. 
        Returns w and Yhat
        """
        w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y)) #Use np.dot() to do matrix multiplication
        Yhat = np.dot(X, w)  #X must be first so the inner dimensions are the same

        #L2 Regularization
        if Ridge:
            w_L2 = np.linalg.solve(L2*np.eye(D) + X.T.dot(X), X.T.dot(Y))  #Solves w and adds the weight penalty
            Yhat_L2 = X.dot(w_L2)
        
        return w, Yhat, w_L2, Yhat_L2

    def MSE(self, Y, Yhat, N):
        """
        Determines the Mean Square of Error, determines how acuare the model is. The lower the better
        """
        delta = Y - Yhat
        mse = delta.dot(delta) / N
        return mse
    
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
        return R
        
    def Predict(self, X_test, w):
        Yhat = np.dot(X_test, w) 
        return Yhat

    def Draw_Result(self, Yhat, Yhat_L2, Y_test):
        i = 0
        while i < len(Yhat):
            predict = plt.scatter(i, Yhat[i], c = '#1f77b4')
            predict_L2 = plt.scatter(i, Yhat_L2[i], c = '#EA320A')
            actual = plt.scatter(i, Y_test[i], c = '#bcbd22')
            i+= 1

        predict.set_label('Predict')
        predict_L2.set_label('Predict L2')
        actual.set_label('Actual')
        plt.legend()
        plt.show()

filename = r"C:\Dev\SourceCode\Personal\mk007---ML-Python-library\data\archive\hou_all.csv"

MultiL = Multiple_Linear()
X_train, Y_train, X_test, Y_test, N, D = MultiL.get_data(filename, test_size = 350)
#X, Y = MultiL.get_data_for_loop()
#MultiL.plot_graph(X, Y)
w, Yhat, w_L2, Yhat_L2 = MultiL.calc_weights(X_train, Y_train, Ridge=True, L2=10, Feat=D)

R1 = MultiL.R_squared(Y_train, Yhat)
R2 = MultiL.R_squared(Y_train, Yhat_L2)
print("The value of R squared is", R1)
print("The value of R squared is", R2)

mse1 = MultiL.MSE(Y_train, Yhat, len(Y_train))
mse2 = MultiL.MSE(Y_train, Yhat_L2, len(Y_train))
print("The value of MSE is", mse1)
print("The value of MSE is", mse2)

result = MultiL.Predict(X_test, w)
result_L2 = MultiL.Predict(X_test, w_L2)

MultiL.Draw_Result(result, result_L2, Y_test)
