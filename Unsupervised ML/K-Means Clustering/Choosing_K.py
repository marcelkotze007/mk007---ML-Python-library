import numpy as np
import matplotlib.pyplot as plt

from K_Means import K_means
from Test_K import create_data

def choosing_k(X):
    model = K_means()
    costs = np.empty(10)
    costs[0] = None
    for k in range(1,10):
        M, R = model.fit(X, K=3, max_iter=20, beta=1, show_cost=False, show_grid = False, validate=False, Y=None, answer = True)
        c = model.cost(X, R, M)
        costs[k] = c

    plt.plot(costs)
    plt.title("Cost vs K")
    plt.show()   

if __name__ == "__main__":
    X = create_data()
    choosing_k(X)
    


