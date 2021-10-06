import numpy as np 
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage

def create_data(D = 2, S = 4, N = 900):
    mu1 = np.array([0,0])
    mu2 = np.array([S,S])
    mu3 = np.array([0,S])

    X = np.zeros((N, D))
    X[:300, :] = np.random.randn(300, D) + mu1
    X[300:600, :] = np.random.randn(300, D) + mu2
    X[600:, :] = np.random.randn(300, D) + mu3

    plt.scatter(X[:,0], X[:, 1])
    plt.show()

    return X

def H(X, name):
    print(name)
    Z = linkage(X, name)
    plt.title(name)
    dendrogram(Z)
    plt.show()

if __name__ == "__main__":
    methods = {1:'ward', 2:'single', 3:'complete'}
    X = create_data()

    for value in methods.values():
        method = value
        H(X, method)