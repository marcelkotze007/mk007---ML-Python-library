import numpy as np 
import matplotlib.pyplot as plt 

import theano
import theano.tensor as T 
from sklearn import shuffle

from Utilities import get_data, get_binary_data, error_rate, relu, init_weights_and_bias

class HiddenLayer(object):
    def __init__(self, M1, M2, an_id):
        self.id = an_id
        self.M1 = M1
        self.M2 = M2
        W, b = init_weights_and_bias(M1, M2)
        self.W = theano.shared(W, 'W_%s' % self.id)
        self.b = theano.shared(b, 'b_%s' % self.id)
        self.params = [self.W, self.b] #This is one list with all the weights and biases
    
    def forward(self, X):
        return relu(X.dot(self.W) + self.b)

class ANN(object):
    def __init__(self, hidden_layer_sizes):
        self.hidden_layer_sizes = hidden_layer_sizes
    
    def fit(self, X, Y, learning_rate=10e-5, reg=0, mu=0.9, decay=0.99, eps=1e-10,
    epochs=400, batch_sz=100, show_fig=False):
        # Allows for running on the GPU
        learning_rate = np.float32(learning_rate)
        mu = np.float32(mu)
        reg = np.float32(reg)
        eps = np.float32(eps)

        # Create a validation set
        X, Y = shuffle(X, Y)
        X = X.astype(np.float32)
        Y = Y.astype(np.int32)
        Xvalid, Yvalid = X[-1000:], Y[-1000:]
        X, Y = X[:-1000], Y[:-1000]

        # Initialize the hidden layers
        N, D = X.shape
        K = len(set(Y)) # Number of classes/number of nodes in final layer
        self.hidden_layers = []
        M1 = D
        count = 0 # Id of the hidden layer
        for M2 in self.hidden_layer_sizes:
            h = HiddenLayer(M1, M2, count)
            self.hidden_layers.append(h)
            M1 = M2 # The output of one layer becomes the input for the next layer
            count += 1
        W, b = init_weights_and_bias(M1, K)
        self.W = theano.shared(W, 'W_logreg')
        self.b = theano.shared(b, 'b_logreg')

        # Collect params for later use
        self.params = [self.W, self.b]
        for h in self.hidden_layers:
            self.params += h.params
        
        # Set up Theano functions and variables




def main():
    X, Y = get_data()

    model = ANN([2000, 1000])
    model.fit(X, Y, show_fig = True)
    print(model.score(X, Y))

if __name__ == "__main__":
    pass