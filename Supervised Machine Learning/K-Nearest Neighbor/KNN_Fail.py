import numpy as np
import matplotlib.pyplot as plt

from KNN import KNN

class knn_fail(object):
    def __init__(self):
        pass
    
    def get_data(self, width = 8, length = 8):
        N = width * length
        X = np.zeros((N, 2))
        Y = np.zeros(N)
        n = 0
        start_t = 0

        #Create alternating pattern
        for i in range(width):
            t = start_t
            for j in range(length):
                X[n] = [i, j] #first iteration will be X[0] = [0, 0], second will be X[1] = [1,0]
                Y[n] = t
                n += 1 #Adds one in each loop
                t = (t + 1) % 2 #Sets each to every second value
            start_t = (start_t + 1) % 2
        
        return X, Y

if __name__ == '__main__':
    KNN_fail = knn_fail()
    X, Y = KNN_fail.get_data()

    plt.scatter(X[:,0], X[:,1], s=100, c=Y, alpha=0.5)
    plt.show()

    model = KNN(3)
    model.fit(X, Y)
    print("Train accuracy:", model.score(X, Y))
