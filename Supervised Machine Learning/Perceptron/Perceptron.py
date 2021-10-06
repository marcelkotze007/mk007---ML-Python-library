import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from future.utils import viewitems

from Util import get_data as get_mnist

def create_data(N = 300, D = 2):
    """
    Simple function to create basic dataset that is linearly separable
    that can be plotted (only has 2 dimensions), used to illistrate how the program works
    """ 
    w = np.array([-0.5, 0.5])
    b = 0.1
    #X is unifromly distributed between -1 and +1
    X = np.random.random((N, D)) * 2 -1
    #Make a prediction for Y given the formula = w.T * x + b. If the formula [>,=,<] 0 ---> predict [1,0,-1] 
    Y = np.sign(X.dot(w) + b) #returns -1, 0, 1 for x<0, x=0, x>0
    return X, Y

def create_simple_xor():
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])
    Y = np.array([0, 1, 1, 0])
    return X, Y

class Perceptron(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y, learning_rate = 0.01, max_epochs = 1000):
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
            #Use the non-zero function to collect all of the samples where Y != Y_hat
            incorrect = np.nonzero(Y != Y_hat)[0] #more detail on nonzero in Basics_own.py
            if len(incorrect) == 0:
                break
            #Choose a random sample from the incorrect samples:
            i = np.random.choice(incorrect) #sets the position of the random incorrect sample = i
            #Then use the update rule to adjust the incorrect x, to either correct or closer to being correct
            print(X[i].shape, Y[i].shape)
            print(Y[i])
            self.w += learning_rate * X[i] * Y[i] #Stochastic gradient decent, se perceptron loss function notes for formula
            self.b += learning_rate * Y[i]

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
        return np.sign(X.dot(self.w) + self.b) #returns -1, 0, 1 for x<0, x=0, x>0

    def score(self, X, Y):
        Y_hat = self.predict(X)
        return np.mean(Y_hat == Y)

if __name__ == "__main__":
    #X, Y = create_data()
    #scatters the data to see what it looks like
    #plt.scatter(X[:, 0], X[:,1], c = Y, s=100, alpha=0.5)
    #plt.show()

    #X, Y = create_simple_xor()

    #MNIST
    X, Y = get_mnist()
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]
    Y[Y == 0] = -1

    N_train = len(Y)//2
    #Remains the same for all the examples of supervised machine learning
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:], Y[N_train:]

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

    