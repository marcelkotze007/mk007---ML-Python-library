import numpy as np
import pandas as pd

class KNN_Util(object):
    def __init__(self):
        pass

    def get_data(self, limit = None, filename = "C:/Users/Marcel/OneDrive/Python Courses/Machine Learning/train.csv"):
        """
        Reads the MNIST dataset and outputs X and Y.
        One can set a limit to the number of rows (number of samples) by editing the 'limit'
        """
        print("Reading in and transforming data...")
        dataset = pd.read_csv(filename).values
        np.random.shuffle(dataset)
        X = dataset[:, 1:] / 255
        Y = dataset[:, 0]
        if limit is not None:
            X, Y = X[:limit], Y[:limit]
        print("Done reading in data...")
        return X, Y
    
    def get_XOR(self, N = 200, D = 2):
        X = np.zeros((N, D))
        X[:N//4] = np.random.random((N//4, D)) / 2 + 0.5 #(0.5 - 1, 0.5 - 1)
        X[N//4:N//2] = np.random.random((N//4, D)) / 2  #(0 - 0.5, 0 - 0.5)
        X[N//2:(N - N//4)] = np.random.random((N//4, D)) / 2 + np.array([[0, 0.5]])  #(0 - 0.5, 0.5 - 1)
        X[(N - N//4):] = np.random.random((N//4, D)) / 2  + np.array([[0.5, 0]])#(0.5 - 1, 0 - 0.5)
        Y = np.array([0]*(N//2) + [1]*(N//2))
        return X, Y

    def get_donut(self, N = 200, R_inner = 5, R_outer = 10):
        """
        Distance from origin is radius + random Gaussian.
        Angle theta is unifomly distributed between (0, 2pi)
        """
        R1 = np.random.rand(N//2) + R_inner
        theta = 2*np.pi*np.random.random(N//2)
        X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T

        R2 = np.random.rand(N//2) + R_outer
        theta = 2*np.pi*np.random.random(N//2)
        X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T

        X = np.concatenate([X_inner, X_outer])
        Y = np.array([0]*(N//2) + [1]*(N//2))
        return X, Y

if __name__ == "__main__":
    KNN_Util = KNN_Util()