import numpy as np 
from Utilities import functions
from ANN_General import ANN
from sklearn.utils import shuffle

fn = functions()
model = ANN(activation_function='relu', GD_method="Batch_SGD")

def main():
    X, Y = fn.get_data(balance_ones = False)

    X, Y = shuffle(X, Y)
    Ntrain = int(0.9 * len(X))
    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    model.fit(Xtrain, Ytrain, epochs=10000, regularization=0.0001, learning_rate=10e-7, M=400, goal='classification',
    costs_show=True, momentum='nesterov', optimizer='rmsprop', early_stop=0.005, batch_size=750)
    print(model.score(Xtest, Ytest))

if __name__ == "__main__":
    main()