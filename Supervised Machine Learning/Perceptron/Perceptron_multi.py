import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from future.utils import viewitems

from Util import get_data as get_mnist

class Perceptron(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y2, learning_rate = 0.01, max_epochs = 1000):
        """
        Takes in arguments X and Y. The learning rate is defined by the user as well as the max_epochs.\n
        learning_rate = 1, 0.1, 0.01, 0.001, 0.0001, ens. \n
        max_epochs is the number of iterations, the higher == higher accuracy
        """
        N, D = X.shape
        #Initialises the w to a random gaussian distributed array
        self.w = np.random.randn(D)
        self.b = 0

        #Creates an empty list to store the costs
        costs = []
        #loop trough all of the epochs
        for epoch in range(max_epochs):
            #First get a prediction to determine which x_values are miss classified
            Y_hat = self.predict(X)
            P = np.argmax(Y_hat, axis = 1)
            #Use the non-zero function to collect all of the samples where Y != Y_hat
            incorrect = np.nonzero(Y2 != P)[0] #more detail on nonzero in Basics_own.py
            if len(incorrect) == 0:
                break
            #Choose a random sample from the incorrect samples:
            i = np.random.choice(incorrect) #sets the position of the random incorrect sample = i
            #Then use the update rule to adjust the incorrect x, to either correct or closer to being correct
            self.w += learning_rate * X[i] * Y2[i].T #Stochastic gradient decent, se perceptron loss function notes for formula
            self.b += learning_rate * Y2[i]

            #The error rate as a % of the number of incorrect/number of samples
            c = len(incorrect)/float(N)
            costs.append(c)
        #Look at some debugging infromation:
        print("Final w: %s Final b: %s Number of epochs: %s / %s" %(self.w, self.b, epoch + 1, max_epochs))

        #Plot the costs to see how we progress through each generation
        plt.plot(costs, label = "costs")
        plt.legend()
        plt.show()

    def predict(self, X):
        #org = np.sign(X.dot(self.w) + self.b) #returns -1, 0, 1 for x<0, x=0, x>0        
        expX = np.exp(X)
        Y_hat = expX / expX.sum(axis = 1, keepdims = True) 
        return Y_hat

    def score(self, X, Y2):
        Y_hat = self.predict(X)
        P = np.argmax(Y_hat, axis = 1)
        return np.mean(P == Y2)

if __name__ == "__main__":

    #MNIST
    X, Y = get_mnist()
    N = len(Y)
    #One-hot encoding:
    Y2 = np.zeros((N, 10))
    Z = np.zeros((N, 10))
    Z[np.arange(N), Y[:].astype(np.int32)] = 1
    #print(Z.shape)
    #print(Y2.shape)
    Y2[:,:] = Z
    #print(Y2.shape)
    #Testing purposes
    #for i in range(20):
    #    print(Y[i], Y2[i])

    N_train = len(Y)//5
    #Remains the same for all the examples of supervised machine learning
    X_train, Y_train = X[:N_train], Y[:N_train]
    #print(Y_train.shape)
    X_train, Y_train = X[:N_train], Y2[:N_train]
    #print(Y_train.shape)
    #print(X_train.shape)
    X_test, Y_test = X[N_train:], Y2[N_train:]

    model = Perceptron()
    t0 = dt.now()
    model.fit(X_train, Y_train, learning_rate=10e-3)
    print("Training time: ", (dt.now() - t0))

    t0 = dt.now()
    print("Traing accuracy: ", model.score(X_train, Y_train))
    print("Computing training accuracy time: ", (dt.now() - t0), "Train size: ", len(Y_train))

    t0 = dt.now()
    print("Testing accuracy: ", model.score(X_test, Y_test))
    print("Computing testing accuracy time: ", (dt.now() - t0), "Test size: ", len(Y_test))     

    