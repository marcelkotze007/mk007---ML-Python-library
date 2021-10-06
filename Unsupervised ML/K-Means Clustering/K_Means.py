import numpy as np
import matplotlib.pyplot as plt

class K_means(object):
    def __init__(self):
        pass
    
    def d(self, u, v):
        diff = u - v
        return diff.dot(diff)

    def cost(self, X, R, M):
        cost = 0
        for k in range(len(M)):
            for n in range(len(X)):
                cost += R[n, k]*self.d(M[k], X[n])
        return cost

    def DBI(self, X, R, M):
        K, _ = X.shape

        sigma = np.zeros(K)
        for k in range(K):
            diffs = X - M[k]
            squared_distances = (diffs * diffs).sum(axis = 1)
            weighted_squared_distances = R[:, k] * squared_distances
            sigma[k] = np.sqrt(weighted_squared_distances).mean()
        
        dbi = 0
        for k in range(K):
            max_ratio = 0
            for j in range(K):
                if k != j:
                    numerator = sigma[k] + sigma[j]
                    denomintor = np.linalg.norm(M[k] - M[j])
                    ratio = numerator/denomintor
                    if ratio > max_ratio:
                        max_ratio = ratio
            dbi += max_ratio
        
        return dbi/K 

    def Purity(self, Y, R):
        N, K = R.shape
        p = 0
        for k in range(K):
            #best_target = -1
            max_intersection = 0
            for j in range(K):
                intersection = R[Y == j, k].sum()
                if intersection > max_intersection:
                    max_intersection = intersection
                    #best_target = j
            p += max_intersection
        return p/N

    def fit(self, X, K = 3, max_iter = 20, beta = 1, show_cost = False, show_grid = False, validate = None, Y = None, answer=False):
        N, D = X.shape
        M = np.zeros((K, D))
        R = np.zeros((N, K))

        for k in range(K):
            M[k] = X[np.random.choice(N)]
        
        if show_grid:
            grid_width = 5
            grid_height = max_iter/grid_width
            random_colors = np.random.random((K, 3))
            plt.figure()

        costs = np.zeros(max_iter)
        for i in range(max_iter):

            if show_grid:
                colors = R.dot(random_colors)
                plt.subplot(grid_width, grid_height, i+1)
                plt.scatter(X[:,0], X[:,1], c = colors)

            for k in range(K):
                for n in range(N):
                    R[n,k] = np.exp(-beta*self.d(M[k], X[n]))/np.sum(np.exp(-beta*self.d(M[j], X[n])) for j in range(K)) 
            
            for k in range(K):
                M[k] = R[:,k].dot(X) / R[:,k].sum()

            costs[i] = self.cost(X, R, M)

            if i > 0:
                if np.abs(costs[i] - costs[i-1]) < 0.1:
                    break
        
        if validate == "DBI":
            dbi = self.DBI(X, R, M)
            print("DBI (Lower is better):", dbi)
        elif validate == "Purity":
            #if Y == None:
            #    print("Please provide the true labels")
            #else:
            purity = self.Purity(Y, R)
            print("Purity (Higher is better):", purity)

        if show_grid:
            plt.show()

        if show_cost:
            plt.plot(costs)
            plt.title("Costs")
            plt.show()

        if answer == True:
            return M, R
        
if __name__ == "__main__":
    model = K_means()