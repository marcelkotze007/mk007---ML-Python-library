import numpy as np
from Pre_proccessing_Data import proccessing_data

#filename = "C:/Users/Marcel/OneDrive/Python Courses/Deep Learning/Commerce Project/ecommerce_data.csv"
PD = proccessing_data()

class logistic_predict:
    def calc_w(self, filename):
        X, Y = PD.get_binary_data(filename)

        D = X.shape[1] #Extracts the amount of features in the array
        W = np.random.randn(D) #Initialize random weights
        b = 0  #Is the bias term and as such is a scalar
         
        return X, Y, D, W, b
    
    def sigmoid(self, z):
        """
        The sigmoid formula gives the probability of a y = 1 for a given value of x
        """
        Sigmoid = 1 / (1 + np.exp(-z))
        return Sigmoid

    def forward(self, X, W, b):
        """
        Uses the sigmoid function by create values z and inserting them into the sigmoid function
        """
        Forward = LP.sigmoid(X.dot(W) + b)
        return Forward
    
    def classification_rate(self, Y, P):
        return np.mean(Y == P)   #This function returns 1 and 0
        #Thus, divide the total number of correct by the total number


LP = logistic_predict()
#X, Y, D, W, b = LP.calc_w(filename)

#P_Y_given_X = LP.forward(X, W, b)
#predictions = np.round(P_Y_given_X) 

#Score = LP.classification_rate(Y, predictions)

#print("Score: ", Score)