import numpy as np
import matplotlib.pyplot as plt

#Alot of the code is explained in further detail in other files, i.e. Cross_Entropy and Logistic_Regr

class gradient_decent_logistic:
    def create_data(self, N = 100, D = 2):
        """
        Creates dataset with a 100 samples with 2 features, that is Gaussian distributed
        """

        X = np.random.randn(N, D)

        #Now set the first 50 values around mean of -2 and the second 50 values to around +2
        X[:50, :] = X[:50, :] - 2 * np.ones((50, D))
        X[50:, :] = X[50:, :] + 2 * np.ones((50, D))

        #Sets the target, first 50 to 0, second 50 to 1
        T = np.array([0] * 50 + [1] * 50)

        #Creates 1 row of 1 of length 100, the transpose to get column
        #shape = 1xN, becomes: Nx1
        ones = np.ones((N, 1))

        #Can then concatenate to create the bias term:
        X_bias = np.concatenate((ones, X), axis = 1)

        #We then randomly create the weights
        w = np.random.randn(D + 1) #There are 3 weights, one for the bais and one for X1, X2 respectively

        return X_bias, w, N, T

    def Dot_of_X_and_w(self, X_bias, w):
        """
        Calculate the dot product between each row of x and w
        Use the dot function to do matrix multiplication
        """
        z = X_bias.dot(w) #This will give an Nx1 Vector as the inner product is: NxD * Dx1 = Nx1

        return z
    
    def sigmoid(self, z):
        #Remember that sigmoid is equal to 1/(1 + e^(-z))
        sig = 1/(1 + np.exp(-z))
        
        return sig

    def cross_entropy(self, T, sig):
        #Remember to add the 1e-10 to the sig, so there is no divide by 0 error
        Error = -1*((T * np.log(sig + 1e-10)) + ((1 - T) * np.log(1 - sig + 1e-10)))
        sum_error = np.sum(Error)
        
        return sum_error

    def closed_form_solution(self, X_bias, T, N):
        """
        Use the closed form solution to calculate the cross-entropy error, can only be done cause the variance
        is the same for each class as it is Gaussain distributed data, that has a default variance of 1.
        Must also first calculate the weights (w) for the data and the bias (b)
        The closed_form_solution function demonstrates that the error is low if the correct values of w is used
        as they were calculated and not randomly selected 
        """
        #Must first calculate the weights, w = (4, 4). Bias, b = 0
        w = np.array([0, 4, 4])

        z = X_bias.dot(w)
        Y_hat = GDL.sigmoid(z)
        E = GDL.cross_entropy(T, Y_hat)
        
        return E       

    def gradient_decent(self, X_bias, T, sig, w, learning_rate = 0.1):
        for i in range(100):
            #Prints out every 10 steps
            if i % 10 == 0:
                print(GDL.cross_entropy(T, sig))

            #The decrease in derivative without the use of a for loop
            w += learning_rate * X_bias.T.dot(T - sig)
            sig = GDL.sigmoid(X_bias.dot(w))    
        
        return w, sig

    def plot_graph(self, X_bias, w):
        plt.scatter(X_bias[:, 1], X_bias[:, 2], c = T, s = 100, alpha = 0.5)
        x_axis = np.linspace(-6, 6, 100)
        y_axis = -1 * (w[0] + x_axis * w[1]) / w[2]
        plt.plot(x_axis, y_axis)
        plt.show()

GDL = gradient_decent_logistic()
X_bias, w, N, T = GDL.create_data() 
z = GDL.Dot_of_X_and_w(X_bias, w)
sig = GDL.sigmoid(z)
sum_error = GDL.cross_entropy(T, sig)
cfs = GDL.closed_form_solution(X_bias, T, N)
adj_w, adj_Y_hat = GDL.gradient_decent(X_bias, T, sig, w)

print("closed form solution: ", cfs)
print("Learning w: ", adj_w)
#print("Adjusted predictions: ", adj_Y_hat)

GDL.plot_graph(X_bias, w)
