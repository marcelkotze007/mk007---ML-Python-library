import numpy as np 
import matplotlib.pyplot as plt 

from scipy.stats import multivariate_normal

def create_data(D = 2, S = 4, N = 2000):
    mu1 = np.array([0,0])
    mu2 = np.array([S,S])
    mu3 = np.array([0,S])

    X = np.zeros((N, D))
    X[:1200, :] = np.random.randn(1200, D)*2 + mu1
    X[1200:1800, :] = np.random.randn(600, D) + mu2
    X[1800:, :] = np.random.randn(200, D)*0.5 + mu3

    plt.scatter(X[:,0], X[:, 1])
    plt.show()

    return X

class GMM(object):
    def __init__(self):
        pass
    
    def fit(self, X, K=3, max_iter=2000, smoothing=1e-2, show_cost=True, show_data=False, debug=True):
        N, D  = X.shape
        M = np.zeros((K,D)) #Means
        R = np.zeros((N,K)) #Responsibilities
        C = np.zeros((K,D,D)) #Covariance matrix --> 3D because cov is already 2D
        pi = np.ones(K)/K

        #Initialize M to random, initialize C to sperical with variance 1
        for k in range(K):
            M[k] = X[np.random.choice(N)]
            C[k] = np.eye(D)
        
        costs = []
        weighted_pdfs = np.zeros((N, K)) #Stores the pdf values so do not need to calculate multiple times
        
        for i in range(max_iter):
            #Step 1 --> Determine assigments/responsibilities
            #The slow method
            #    for k in range(K):
            #        for n in range(N):
            #            weighted_pdfs[n,k] = pi[k] * multivariate_normal.pdf(X[n], M[k], C[k])
            
            #    for k in range(K):
            #        for n in range(N):
            #            R[n,k] = weighted_pdfs[n,k] / weighted_pdfs[n,:].sum()

            #The fast method using vectors:
            for k in range(K):
                weighted_pdfs[:,k] = pi[k]*multivariate_normal.pdf(X, M[k], C[k])
            R = weighted_pdfs / weighted_pdfs.sum(axis=1, keepdims=True)

            #Step 2 --> Recalculate Parameters
            for k in range(K):
                Nk = R[:,k].sum()
                pi[k] = Nk/N
                M[k] = R[:,k].dot(X) / Nk

                ##Faster:
                delta = X - M[k] #NxD matrix
                Rdelta = np.expand_dims(R[:,k], -1) * delta #Multiplies R[:,k] by each col. of delta --> NxD
                C[k] = Rdelta.T.dot(delta) / Nk + np.eye(D) * smoothing #DxD matrix
                ##Slower
                #C[k] = np.sum(R[n,k]*np.outer(X[n]-M[k], X[n]-M[k]) for n in range(N))/Nk + np.eye(D)*smoothing

            cost = np.log(weighted_pdfs.sum(axis=1)).sum()
            costs.append(cost)
            if i > 0:
                if np.abs(costs[i] - costs[i-1]) < 0.1:
                    break

        if show_cost:
            plt.plot(costs)
            plt.title("Log-likelihood")
            plt.show()

        if show_data:
            random_colors = np.random.random((K, 3))
            colors = R.dot(random_colors)
            plt.scatter(X[:,0], X[:,1], c=colors)
            plt.show()
        
        if debug:
            print("pi:", pi)
            print("means:", M)
            print("Covariance:", C)
        
        return R

if __name__ == "__main__":
    X = create_data()
    
    model = GMM()
    model.fit(X, K=3, max_iter=20, smoothing=1e-2, show_cost=True, show_data=True, debug=True)

    