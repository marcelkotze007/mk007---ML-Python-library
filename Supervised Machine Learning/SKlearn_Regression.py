import numpy as np
from sklearn import tree
from sklearn import neighbors
import matplotlib.pyplot as plt

def create_data(N = 200, N_train = 20):
    X = np.linspace(0, 10 , N).reshape(N,1) #SKlearn expects an NxD matrix, thus need to create a D value
    Y = np.sin(X)

    return X, Y

if __name__ == "__main__":
    N = 200
    N_train = 20
    X, Y = create_data(N= N, N_train= N_train)

    #Randomly select a point
    index = np.random.choice(N, N_train)
    X_train = X[index]
    Y_train = Y[index]

    knn_model = neighbors.KNeighborsRegressor(n_neighbors=3, weights='distance', algorithm='kd_tree')
    knn_model.fit(X_train, Y_train)
    Y_hat_knn = knn_model.predict(X)

    tree_model = tree.DecisionTreeRegressor(max_depth=5)
    tree_model.fit(X, Y)
    Y_hat_dt = tree_model.predict(X)

    plt.scatter(X_train, Y_train)
    plt.plot(X, Y)
    plt.plot(X, Y_hat_knn, label = "KNN")
    plt.plot(X, Y_hat_dt, label = "Decision Tree")
    plt.legend()
    plt.show()

    