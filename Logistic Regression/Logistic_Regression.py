import numpy as np
import matplotlib.pyplot as plt

class simple_Logistic:
    def create_data(self):
        N = 100
        D = 2

        X = np.random.randn(N, D)
        
        #Adding the bias term:
        #Add a column of ones, and include the bias in the weights w
        #need the array to have N rows and 1 column and such create the ones as follows:
        ones =  np.array([[1] * N]).T

        #Concatenate the ones to the orginal data
        X_bias = np.concatenate((ones, X), axis = 1)

        #Now randomly create a weight vector
        w = np.random.randn(D + 1) #Remember that it will be W.Tdot(X) so the inner dimensions must be the same
        #Values do not matter, just want to calculate the sigmoid

        return X_bias, w

    def Dot_of_X_and_w(self, X_bias, w):
        """
        Calculate the dot product between each row of x and w
        Use the dot function to do matrix multiplication
        """
        z = X_bias.dot(w) #This will give an Nx1 Vector as the inner product is: NxD * Dx1 = Nx1

        return z
    
    def sigmoid(self, z):
        #Remember that sigmoid is equal to 1/(1 + e^(-z))
        Sigmoid = 1/(1 + np.exp(-z))
        
        return Sigmoid

SL = simple_Logistic()
X_bias, w = SL.create_data()
z = SL.Dot_of_X_and_w(X_bias, w)
Sigmoid = SL.sigmoid(z)

print(Sigmoid)

plt.scatter(X_bias[:, 1], Sigmoid)
plt.show()

