import numpy as np
import matplotlib as plt
from datetime import datetime as dt
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn
from Util import get_data
from future.utils import viewitems

class naive_bayes(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y, smoothing = 10e-3):
        self.gaussians = dict()
        self.priors = dict()
        labels = set(Y)
        for c in labels:
            current_x = X[Y == c]
            self.gaussians[c] ={
                'mean': current_x.mean(axis = 0),
                'var': current_x.var(axis = 0) + smoothing,
            } 
            self.priors[c] = float(np.log(len(Y[Y == c])) - np.log(len(Y)))

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(Y == P)
    
    def predict(self, X):
        N = X.shape[0]
        K = len(self.gaussians)
        P = np.zeros((N, K))
        for c, g in viewitems(self.gaussians):
            mean, var = g['mean'], g['var']
            P[:,c] = mvn.logpdf(X, mean=mean, cov=var) + self.priors[c]
        return np.argmax(P, axis=1)

if __name__ == "__main__":
    X, Y = get_data(limit = None)
    
    N_train= len(Y)//2
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:], Y[N_train:]

    model = naive_bayes()
    t0 = dt.now()
    model.fit(X_train, Y_train)
    print("Training time: ", (dt.now() - t0))

    t0 = dt.now()
    print("Traing accuracy: ", model.score(X_train, Y_train))
    print("Computing training accuracy time: ", (dt.now() - t0), "Train size: ", len(Y_train))

    t0 = dt.now()
    print("Testing accuracy: ", model.score(X_test, Y_test))
    print("Computing testing accuracy time: ", (dt.now() - t0), "Test size: ", len(Y_test))