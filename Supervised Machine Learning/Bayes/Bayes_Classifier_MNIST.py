import numpy as np 
import pickle
from datetime import datetime as dt 
from scipy.stats import norm
from scipy.stats import multivariate_normal as mvn 
from future.utils import viewitems

from Util import get_data

class bayes(object):
    def __init__(self):
        pass
    
    def fit(self, X, Y, smoothing = 10e-3):
        D = X.shape[1]
        self.gaussians = dict() #Create an empty dictionary for the gaussian adjusted values
        self.priors = dict()
        labels = set(Y) #Sets labels equal to Y
        #Create for loop to itterate through the 0-9 labels:
        for c in labels: #C represents class
            current_x = X[Y == c]
            self.gaussians[c] = {
                'mean' : current_x.mean(axis = 0), #Creates a mean that stores the mean for a certain class
                #Creates the cov for a class, aswell as adding an identity matrix * the smoothing
                'cov' : np.cov(current_x.T) + np.eye(D) * smoothing,
            }
            #priors is just p(C), thus counts how many times c==Y / number of Y
            self.priors[c] = float(np.log(len(Y[Y == c])) - np.log(len(Y))) #use log to speed up the calculations
    
    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)

    def predict(self, X):
        N = X.shape[0]
        K = len(self.gaussians) #determines the length of the dictonary
        P = np.zeros((N, K)) #Creates an empty array with the correct dimensions
        for c, g in viewitems(self.gaussians):
            mean, cov = g['mean'], g['cov']
            P[:, c] = mvn.logpdf(X, mean=mean, cov=cov) + self.priors[c]
        return np.argmax(P, axis=1) #Returns the max value along a given axis
    
    def cross_validate(self, X, Y, K = 5):
        """
        Uses the K-Fold Cross-Validation method
        """
        scores = []
        size = len(Y)//K
        for i in range(K):
            X_valid, Y_valid = X[i*size: (i+1)*size], Y[i*size: (i+1)*size]
            X_train = np.concatenate((X[:i*size], X[(i+1)*size:]))
            Y_train = np.concatenate((Y[:i*size], Y[(i+1)*size:]))

            model.fit(X_train, Y_train)
            cv_score = model.score(X_valid, Y_valid)
            scores.append(cv_score)
            for j in range(len(scores)):
                if cv_score > scores[j]:
                    with open ('best_model.pkl', 'wb') as afile:
                        pickle.dump(model, afile)            
        return scores

if __name__ == "__main__":
    X, Y = get_data()

    N_train = len(Y)//2
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:], Y[N_train:]

    model = bayes()
    t0 = dt.now()
    model.fit(X_train, Y_train)
    print("Training time: ", (dt.now() - t0))

    t0 = dt.now()
    print("Traing accuracy: ", model.score(X_train, Y_train))
    print("Computing training accuracy time: ", (dt.now() - t0), "Train size: ", len(Y_train))

    t0 = dt.now()
    print("Testing accuracy: ", model.score(X_test, Y_test))
    print("Computing testing accuracy time: ", (dt.now() - t0), "Test size: ", len(Y_test)) 

    #t0 = dt.now()
    #print("K-Fold Cross Validation: ", model.cross_validate(X, Y, K = 5))
    #print("Computing testing accuracy time: ", (dt.now() - t0), "Test size: ", len(Y_test)) 
      