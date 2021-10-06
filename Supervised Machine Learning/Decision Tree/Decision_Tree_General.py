import numpy as np 
from datetime import datetime as dt 
from sklearn.utils import shuffle

from Util import get_data, get_XOR, get_donut

def entropy(y):
    """
    The Entropy Function. Assume binary output: P(X = 1) = p, P(X = 0) = (1 - p). Returns the entropy of P.
    Entropy measures how much information we we get from finding out the value of the RV
    """
    #Assume y is binary - 0 or 1
    N = len(y) #Measures the len of array Y
    s1 = (y==1).sum() #Counts the values that are equal to 1
    if 0 == s1 or N == s1: # If there is no result, return 0
        return 0
    p1 = float(s1) / N  #number of values of y=1 / total amount of values
    p0 = 1- p1
    return -p0 * np.log2(p0) - p1 * np.log2(p1) #Actual entropy formula

class TreeNode:
    def __init__(self, depth = 0, max_depth = None):
        self.depth = depth #The depth of the current if loop
        self.max_depth = max_depth  #The max amount of splits allowed and the deepest if loop

    def fit(self, X, Y):
        #Base case
        #Look if there is only one label or the len of Y = 1, cannot split and return this label as prediction
        if len(Y) == 1 or len(set(Y)) == 1:
            self.col = None
            self.split = None
            self.left = None
            self.right = None
            self.prediction = Y[0]
        else:
            D = X.shape[1] #The number of dimensions in X
            cols = range(D) #The col we want to loop through are just 0 - D

            max_IG = 0
            best_col = None
            best_split = None
            for col in cols: #Look for the best column, by looping through all the columns
                IG, split = self.find_split(X, Y, col) #Will find the best split, given a certain column
                if IG > max_IG: #So if this is larger than our current IG, we set the new IG to max_IG
                    max_IG = IG
                    best_col = col
                    best_split = split
            #Another base case where we check if max_IG = 0, which means cannot split further and predict
            #the current value, make this a leaf node
            if max_IG == 0:
                self.col = None
                self.split = None
                self.left = None
                self.right = None
                self.prediction = np.round(Y.mean())
            else: 
                self.col = best_col     #Keep track of our best col and best split
                self.split = best_split #Saves the values created by self.find_split()
                #Final base case, that if reached will not split anymore, set the max depth
                if self.depth == self.max_depth:
                    self.left = None
                    self.right = None
                    self.prediction = [ #There are 2 predictions, one for left and one for right
                        np.round(Y[X[:,best_col] < self.split].mean()), #Takes the majority class after splitting data
                        np.round(Y[X[:, best_col] >= self.split].mean()), #Use majority to make predictions
                    ]                                                     #If majority: Y=1, predict one
                else: #If not in a base case we should do recursion
                    left_idx = (X[:, best_col] < best_split) #All of the points where best_col is less than the split
                    X_left = X[left_idx] #All the X values where less than split
                    Y_left = Y[left_idx]
                    self.left = TreeNode(self.depth + 1, self.max_depth) #Create a new tree node on the left side
                    #Think of this as if P< 0.5, then go to the left outcome, this is where we move to the next level
                    self.left.fit(X_left, Y_left) #Calls the fit function again and loops through the same sequence again

                    right_idx = (X[:, best_col] >= best_split) #All of the points where this attribute is less than split 
                    X_right = X[right_idx] #All the X values where more than split
                    Y_right = Y[right_idx]
                    self.right = TreeNode(self.depth +1, self.max_depth) #Create the right child
                    #Think of this as if P >= 0.5 go to the right outcome, move to the next level of if statements
                    self.right.fit(X_right, Y_right)#Goes through the same sequence again until a base case is reached
    
    def find_split(self, X, Y, col):
        """
        Finds the best split given a certain column. Returns the IG (information gain) and the point,
        at which to split the column
        """
        x_values = X[:,col]
        sort_idx = np.argsort(x_values) #Sorts the indexes
        x_values = x_values[sort_idx]
        y_values = Y[sort_idx] #Sort the y values in the same way as the x-values

        boundaries = np.nonzero(y_values[:-1] != y_values[1:])[0] #This is where the label change from 0 to 1
        #in np.nonzero(The first one starts at the bottom, the other one starts at the top and meet in the middle)
        #We shift Y values by one and then check if the current value is not equal to the next value
        best_split = None
        max_IG = 0
        for i in boundaries: #Gives us the index of the boundries
            split = (x_values[i] + x_values[i + 1]) / 2 #Add to values / 2, splits it in the middle
            IG = self.information_gain(x_values, y_values, split) #Check the IG gain from this split
            if IG > max_IG:
                max_IG = IG
                best_split = split
        return max_IG, best_split #The values computed is then fed back into the recursion and compared with the 
                                  #stored values, if larger, then they are stored
    
    def information_gain(self, x, y, split):
        """
        Takes in arguments x, y, split and returns Information gain.\n
        The formula: IG(Y | split on X) = H(Y) - p(Y=1) * H(Y_left) - P(Y=0) * H(Y_right).\n
        H(Y) - Is the information entropy of Y, usually H(Y) = 1 for binomial \n
        p(Y=1) = Number of times Y = 1 / Number of data \n
        p(Y=0) = Number of times Y = 0 / Number of data \n
        H(Y_left) = The information entropy gained from left child \n
        H(Y_right) = The entropy of right child \n
        """
        y0 = y[x < split] #any Y that x is less than split
        y1 = y[x >= split]
        N = len(y)
        y0_len = len(y0)
        if y0_len == 0 or y0_len == N: #Information gain will be 0
            return 0
        p0 = float(len(y0)) / N
        p1 = 1 - p0
        return entropy(y) - p0*entropy(y0) - p1*entropy(y1) #Calls the entropy function
    
    def predict_one(self, X):
        """
        Checks for the base cases. So if self.col and self.split != None, then a split occured 
        """
        if self.col is not None and self.split is not None:
            feature = X[self.col]
            if feature < self.split: #moves to the left side
                if self.left: #Check if there is a recursive child
                    p = self.left.predict_one(X) #Call the function again
                else:
                    p = self.prediction[0] #returns this leaf nodes prediction
            else:
                if self.right: #If we dont go to the left we go to the right
                    p = self.right.predict_one(X) #check if there is a right child, and call function again
                else:
                    p = self.prediction[1] #If leaf node return prediction directly
        else:
            p = self.prediction #Last base case, if leaf node then return prediction
        return p
    
    def predict(self, X):
        """
        Calls predict_one function for each indivual X 
        """
        N = len(X)
        P = np.zeros(N)
        for i in range(N):
            P[i] = self.predict_one(X[i])
        return P

class DecisionTree:
    """
    This is a rapper class. TreeNode will still do all the work, will just call it from this class
    """
    def __init__(self, max_depth = None):
        self.max_depth = max_depth
    
    def fit(self, X, Y):
        self.root = TreeNode(max_depth=self.max_depth)
        self.root.fit(X, Y)
    
    def predict(self, X):
        return self.root.predict(X)

    def score(self, X, Y):
        P = self.predict(X)
        return np.mean(P == Y)


if __name__ == "__main__": #All this stays mostly the same
    X, Y = get_data()
    #The fastest way of extracting only the rows with label 0 or 1
    idx = np.logical_or(Y == 0, Y == 1)
    X = X[idx]
    Y = Y[idx]

    #X, Y = get_donut()
    #X, Y = get_XOR()

    X, Y = shuffle(X, Y)

    N_train = len(Y)//2
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:], Y[N_train:]

    model = DecisionTree()
    t0 = dt.now()
    model.fit(X_train, Y_train)
    print("Training time: ", (dt.now() - t0))

    t0 = dt.now()
    print("Traing accuracy: ", model.score(X_train, Y_train))
    print("Computing training accuracy time: ", (dt.now() - t0), "Train size: ", len(Y_train))

    t0 = dt.now()
    print("Testing accuracy: ", model.score(X_test, Y_test))
    print("Computing testing accuracy time: ", (dt.now() - t0), "Test size: ", len(Y_test)) 
