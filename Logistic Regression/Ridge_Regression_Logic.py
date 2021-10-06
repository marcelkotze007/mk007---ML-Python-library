import numpy as np

#look at other files in folder for more detail on the code

class ridge_regression:
    def create_data(self, N = 100, D = 2):
        X = np.random.randn(N, D)

        #Center the first 50 points at (-2, -2)
        X[:50,:] = X[:50,:] - 2*np.ones((50, D))
        #Center the second 50 points at (2, 2)
        X[50:,:] = X[50:,:] + 2*np.ones((50, D))

        #Also known as Y
        T = np.array([0] * 50 + [1] * 50)

        #add bias:
        ones = np.ones((N,1))
        X_bias = np.concatenate((ones, X), axis = 1)

        return X_bias, T

    def sig_active(self, D = 2):
        #call function:
        X_bias, T = RR.create_data()
        
        #set w to random values:
        w = np.random.randn(D+1)

        #sigmoid activation:
        z = X_bias.dot(w)

        return z, w, X_bias, T

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy_error(self, T, Y):
        return -np.mean(T * np.log(Y) + (1 - T) * np.log(1 - Y))
    
    def gradient_decent(self, learning_rate = 0.01):
        z, w, X_bias, T = RR.sig_active()
        Y = RR.sigmoid(z)

        for i in range(100):
            if i % 10 == 0:
                print(RR.cross_entropy_error(T, Y))

            #The only addition is the lamda*weight, to add a penalty to the weight
            w += learning_rate * (X_bias.T.dot(T - Y) - 0.1*w)
            Y = RR.sigmoid(X_bias.dot(w))
        
        print("Final w: ", w)

RR = ridge_regression()

RR.gradient_decent()



