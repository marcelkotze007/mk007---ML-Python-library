import numpy as np
import matplotlib.pyplot as plt

from K_Means import K_means

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

if __name__ == "__main__":
    
    X = create_data()

    model = K_means()
    model.fit(X, K=5, max_iter=15,beta=0.3,show_cost=True, show_grid=True, validate="DBI", Y=None)
    