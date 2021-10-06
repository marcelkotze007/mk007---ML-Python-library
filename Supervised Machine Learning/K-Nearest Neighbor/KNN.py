import numpy as np
from sortedcontainers import SortedList
from future.utils import viewitems
import matplotlib.pyplot as plt

from datetime import datetime as dt
from Utilities_KNN import KNN_Util

class KNN(object):
    """
    Takes in argument k which is a hyper parameter
    """
    def __init__(self, k):
        self.k = k

    def fit(self, X, Y):
        """
        This is equal to train. This is a lazy classifier and as such all it does is saves X and saves Y
        """
        self.X = X
        self.Y = Y

    def predict(self, X):
        """
        Predicts the inputs
        """
        Y = np.zeros(len(X))

        #use enumerate to create a list, with a value
        for i, x in enumerate(X):
            #The load sets how large the sorted list should be
            """
            sortedlist = SortedList()
            #For each input test point. Loop trough the training data = self.X
            for j, x_train in enumerate(self.X):
                diff = x - x_train
                #Use the square distance as it is the same as the Euclidean Distance
                d = diff.dot(diff)
                #If the sorted list is smaller than size k, just add the value to the list
                if len(sortedlist) < self.k:
                    #Add the current point without checking anything
                    sortedlist.add((d , self.Y[j]))
                else:
                    #Checks the value at the end, as it will be the biggest distance
                    if d < sortedlist[-1][0]:
                        #If the current value is less than that, del the value and add the current value
                        del sortedlist[-1]
                        sortedlist.add((d, self.Y[j]))
            """        
            #Elimantes the second for loop
            N = self.X.shape[0] # measures the length of X and sets it to N
            diff = self.X - np.array([x]).repeat(N,0)
            d = np.sum(diff**2, axis = 1) 
            min_idxs = d.argpartition(self.k)[:self.k]
            sortedlist = [(d[i], self.Y[i]) for i in min_idxs]
            
            #Create an empty dictionary to collect all of the votes
            votes = {}
            #Loop through the sortedlist of KNN
            for _, v in sortedlist: #Only care about the second elimante as that is the class
                votes[v] = votes.get(v, 0) + 1 #Votes is the count to the key, and key becomes the value
            max_votes = 0
            max_votes_class = -1
            
            #The viewitems fucntion changes as the dict changes and does not need to be reinitialised every time
            for v, count in viewitems(votes):
                if count > max_votes:
                    #If current votes is larger than the max votes, make it the max votes
                    max_votes = count
                    max_votes_class = v
            Y[i] = max_votes_class
        return Y

    def score(self, X, Y):
        """
        Makes a prediction on X and returns the accuracy. Same format and result as Skit-Learn
        """
        Y_hat = self.predict(X)
        return np.mean(Y_hat == Y)  #Returns an array of true and false, thus 0 and 1, which then takes the sum
                                    #And divides by N

if __name__ == '__main__':
    KNN_Util = KNN_Util()
    X, Y = KNN_Util.get_data(limit=2000)
    Ntrain = len(X)//2 #limit to 2000 data points, the first 1000 is the train and the second 1000 is the test
    X_train, Y_train = X[:Ntrain], Y[:Ntrain]
    X_test, Y_test = X[Ntrain:], Y[Ntrain:]
    train_scores = []
    test_scores = []
    #Test for different values of k to see which value gives the best score:
    #K is a hyper parameter and as such has to be dicided beforehad
    kset = (1, 2, 3, 4, 5, 10)
    for k in kset:
        print("\nk = ", k)
        model = KNN(k)
        t0 = dt.now()
        model.fit(X, Y)
        print("Training time: ", (dt.now() - t0))

        print("Start Training")
        t0 = dt.now()
        train_score = model.score(X_train, Y_train)
        train_scores.append(train_score)
        print("Train accuracy: ", train_score)
        print("Time to compute training accuracy: ", (dt.now() - t0), "Train size: ", len(Y_train))

        print("Start Testing")
        t0 = dt.now()
        test_score = model.score(X_test, Y_test)
        test_scores.append(test_score)
        print("Test accuracy: ", test_score)
        print("Time to compute testing accuracy: ", (dt.now() - t0), "Test size: ", len(Y_test))

"""
plt.plot(kset, train_scores, label = "Train Scores")
plt.plot(kset, test_scores, label = "Test Scores")
plt.legend()
plt.show()
"""
