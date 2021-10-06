import numpy as np
import matplotlib.pyplot as plt

class cross_entropy_error:
    def create_data(self):
        
        N = 100
        D = 2

        X = np.random.randn(N, D)

        #Create labels so we can calculate the error
        #Set the first 50 points to be centred at X = -2, Y = -2
        X[:50, :] = X[:50, :] - 2 * np.ones((50, D)) #This basically subtracts 2 from the Gaussian distribution
        #Set the second 50 points to be centred at X = 2, Y = 2
        X[50:, :] = X[50:, :] + 2 * np.ones((50, D))  #This adds 2 to the Gaussian distribution

        #Next is to create an array of targets, witht the first 50 = 0,and the second 50 = 1
        T = np.array([0] * 50 + [1] * 50)

        #For more info on the addition of ones and concatenate, see Logistic_Regression.py
        ones = np.array([[1] * N]).T
        X_bias = np.concatenate((ones, X), axis = 1)

        #Creates random Gaussian distributed weights 
        w = np.random.randn(D + 1) #The D+1 is for the extra column of ones (bias) added

        return X_bias, w, T, N
    
    #The following function are also from Logistic_Regression.py
    def Dot_of_X_and_w(self, X_bias, w):
        """
        Calculate the dot product between each row of x and w
        Use the dot function to do matrix multiplication
        """
        z = X_bias.dot(w) #This will give an Nx1 Vector as the inner product is: NxD * Dx1 = Nx1

        return z
    
    def sigmoid(self, z):
        #Remember that sigmoid is equal to 1/(1 + e^(-z))
        Y_hat = 1/(1 + np.exp(-z))
        #Sigmoid is also be equal to Y_hat the predicted outcome
        return Y_hat

    def cross_entropy(self, T, Y_hat, N):
        #E = 0
        #for i in range(N):
        #    if T[i] == 1:
                #If the target is == 1, then we subtract the ([t = 1])log(y) as the (1 - [t = 1])log(y) = 0
        #        E -= np.log(Y_hat[i])
        #    else:
                #This is the second part of the function as (1 - [t = 0])log(1 - y) will be used as ([t = 0])log(y) = 0
        #        E -= np.log(1 - Y_hat[i])
        #The E that is returned is the -sum of all the errors
        #Instead of using for loops, which are slow use the following:
        Error = (-1.0 *((T*np.log(Y_hat + 1e-10)) + (1 - T)*np.log(1 - Y_hat)))
        E = np.sum(Error)
        return E

    def closed_from_solution(self, X_bias, T, N):
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
        Y_hat = CEE.sigmoid(z)
        E = CEE.cross_entropy(T, Y_hat, N)
        
        return E


CEE = cross_entropy_error()

X_bias, w, T, N = CEE.create_data()
z = CEE.Dot_of_X_and_w(X_bias, w)
Y_hat = CEE.sigmoid(z)
E = CEE.cross_entropy(T, Y_hat, N)
print(E)

Closed_E = CEE.closed_from_solution(X_bias, T, N)
print("Showcase that error is much smaller if correct w is chosen: ", Closed_E)

#draws all of the points
plt.scatter(X_bias[:,1], X_bias[:, 2], c = T, s = 100, alpha = 0.5 ) #c = color, s = size of the dots, alpha = transparency

#draw the line
x_axis = np.linspace(-6, 6, 100)
y_axis = -1 * x_axis
plt.plot(x_axis, y_axis)
plt.show()