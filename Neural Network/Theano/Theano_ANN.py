import numpy as np 
import matplotlib.pyplot as plt 

import theano.tensor as T 
import theano

from util import get_normalized_data, y2indicator

def error_rate(p, t):
    return np.mean(p != t)

def relu(a):
    return a * (a>0)

def main():
    X_train, X_test, Y_train, Y_test = get_normalized_data()
    max_iter = 20
    print_period = 10
    lr = 10e-6
    reg = 10e-3

    Y_train_ind = y2indicator(Y_train)
    Y_test_ind = y2indicator(Y_test)

    N, D = X_train.shape
    batch_sz = 500
    n_batches = N//batch_sz 

    M = 300
    K = 10
    W1_init = np.random.randn(D, M) / np.sqrt(D)
    b1_init = np.zeros(M)
    W2_init = np.random.randn(M, K) / np.sqrt(M)
    b2_init = np.zeros(K)

    #Create the theano variables and expressions
    #First create placeholder --> defines the type for the variables
    thX = T.matrix('X')
    thT = T.matrix('T')

    #All the parameters will be shared variables
    W1 = theano.shared(W1_init, 'W1')
    b1 = theano.shared(b1_init, 'b1')
    W2 = theano.shared(W2_init, 'W2')
    b2 = theano.shared(b2_init, 'b2')

    #Call the relu and the softmax:
    thZ = relu(thX.dot(W1) + b1)
    thY = T.nnet.softmax(thZ.dot(W2) + b2)

    #cost function
    cost = -(thT * T.log(thY)).sum() + reg*((W1*W1).sum() + (b1*b1).sum() + (W2*W2).sum() + (b2*b2).sum())
    prediction = T.argmax(thY, axis=1)

    update_W1 = W1 - lr*T.grad(cost, W1)
    update_b1 = b1 - lr*T.grad(cost, b1)
    update_W2 = W2 - lr*T.grad(cost, W2)
    update_b2 = b2 - lr*T.grad(cost, b2)

    #Create the train function
    train = theano.function(
        inputs = [thX, thT],
        updates=[(W1, update_W1), (b1, update_b1), (W2, update_W2), (b2, update_b2)]
    )

    get_prediction = theano.function(
        inputs=[thX, thT],
        outputs=[cost, prediction]
    )

    LL = []
    for i in range(max_iter):
        for j in range(n_batches):
            Xbatch = X_train[j*batch_sz:(j*batch_sz + batch_sz),]
            Ybatch = Y_train_ind[j*batch_sz:(j*batch_sz + batch_sz),]

            #Call theonos train function:
            train(Xbatch, Ybatch)

            if j % print_period ==0:
                cost_val, prediction_val = get_prediction(X_test, Y_test_ind)
                err = error_rate(prediction_val, Y_test)
                print("Cost/error at iteration i=%d, j=%d: %.3f / %.3f" %(i, j, cost_val, err))
                LL.append(cost_val)
    
    plt.plot(LL)
    plt.show()

if __name__ == "__main__":
    main()