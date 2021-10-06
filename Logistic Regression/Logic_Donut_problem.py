import numpy as np
import matplotlib.pyplot as plt

#The key to solving this problem is to turn it into a 3D-problem, with radius being the z axis
#Then draw a plane between the 2 datasets

class donut:
    def create_data(self, N = 1000, D = 2):
        #Gonna have 2 radia:
        R_inner = 5
        R_outer = 10

        #The // means that the answer will be an integer and not a float
        R1 = np.random.randn(N//2) + R_inner
        theta = 2*np.pi*np.random.random(N//2)
        X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T #So N goes along the rows

        R2 = np.random.randn(N//2) + R_outer
        theta = 2*np.pi*np.random.random(N//2)
        X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

        X = np.concatenate([X_inner, X_outer])
        Y = np.array([0] * (N//2) + [1] * (N//2))

        #Plot the data to showcase the problem
        #plt.scatter(X[:,0], X[:,1], c = Y)
        #plt.autoscale(enable=False)
        #plt.show()

        #Add the bias term:
        ones = np.ones((N, 1))

        #The key to solving this problem is creating a column that represents the radius of the point
        r = np.zeros((N, 1)) #This allows us to linearly separate the datapoints
        for i in range(N):
            #Calculates the radius
            r[i] = np.sqrt(X[i,:].dot(X[i,:])) 
        
        X_bias = np.concatenate((ones, r, X), axis = 1)

        return X_bias, Y

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def cross_entropy_error(self, Y, Y_hat):
        return -np.mean(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

    def gradient_decent(self, learning_rate = 0.0001):
        X_bias, Y = DP.create_data()

        N, D = X_bias.shape
        w = np.random.randn(D)

        #L2 Regularization
        l2 = 0.1
        
        error = []
        for i in range(5000):
            z = X_bias.dot(w)
            Y_hat = DP.sigmoid(z)
            cost = DP.cross_entropy_error(Y, Y_hat)
            if i % 100 == 0:
                print(i, cost)
            
            error.append(cost)

            #gradient decent:
            w += learning_rate * (X_bias.T.dot(Y - Y_hat) - l2 * w)
        
        plt.plot(error)
        plt.title('Cross-Entropy Error')
        plt.show()

        print("Final w: ", w)
        print("Final classification rates: ", 1 - np.abs(Y - np.round(Y_hat)).sum()/N)


DP = donut()

DP.gradient_decent()
