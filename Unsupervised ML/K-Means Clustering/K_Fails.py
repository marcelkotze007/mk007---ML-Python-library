import numpy as np
import matplotlib.pyplot as plt

from K_Means import K_means

def donut(D = 2, N = 1000, R_inner = 5, R_outer = 10):

    R1 = np.random.randn(N//2) + R_inner
    theta = 2*np.pi*np.random.randn(N//2)
    X_inner = np.concatenate([[R1*np.cos(theta)], [R1*np.sin(theta)]]).T

    R2 = np.random.randn(N//2) + R_outer
    theta = 2*np.pi*np.random.randn(N//2)
    X_outer = np.concatenate([[R2*np.cos(theta)], [R2*np.sin(theta)]]).T

    X = np.vstack((X_inner,X_outer))

    plt.scatter(X[:,0], X[:, 1])
    plt.show()

    return X

def elong(D=2, N=1000):
    X = np.zeros((N,D))
    X[:500,:] = np.random.multivariate_normal([0,0], [[1,0], [0,20]], 500)
    X[500:,:] = np.random.multivariate_normal([5,0], [[1,0], [0,20]], 500)

    plt.scatter(X[:,0], X[:, 1])
    plt.show()

    return X

def size_dif(D=2,N=1000):
    X = np.zeros((N, D))
    X[:950,:] = np.array([0,0]) + np.random.randn(950,D)
    X[950:,:] = np.array([3,0]) + np.random.randn(50,D)
    
    plt.scatter(X[:,0], X[:, 1])
    plt.show()

    return X

if __name__ == "__main__":
    
    #X = donut()
    #X = elong()
    X = size_dif()

    model = K_means()
    model.fit(X, K=2, max_iter=20,beta=1,show_cost=True, show_grid=True)