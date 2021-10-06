import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

from ANN_General import ANN

def data_regres():
    # generate and plot the data
    N = 500
    X = np.random.random((N, 2))*4 - 2 # in between (-2, +2)
    Y = X[:,0]*X[:,1] # makes a saddle shape
    # note: in this script "Y" will be the target,
    #       "Yhat" will be prediction
    return X, Y

def data_class(N = 500):
    #first cloud is centred at (0, -2)
    X1 = np.random.randn(N, 2) + np.array([0, -2])
    #second cloud is centred at (2, 2)
    X2 = np.random.randn(N, 2) + np.array([2, 2])
    #Third cloud is centred at (-2, 2)
    X3 = np.random.randn(N, 2) + np.array([-2, 2])
    X = np.vstack((X1,X2,X3))
    Y = np.array([0]*N + [1]*N + [2]*N)
    return X, Y

if __name__ == "__main__":
    model = ANN(activation_function='tanh')

    X, Y = data_regres()
    #X, Y = data_class()
    #t0 = dt.now()
    model.fit(X, Y, epochs=10000, regularization=0.005, learning_rate=0.0001, M=30, goal="regression")
    #print(model.score(X, Y))
    #print(dt.now() - t0)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:,0], X[:,1], Y)
    
    line = np.linspace(-2, 2, 20)
    xx, yy = np.meshgrid(line, line)
    Xgrid = np.vstack((xx.flatten(), yy.flatten())).T
    Yhat = model.predict(Xgrid)
    ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], Yhat, linewidth=0.2, antialiased=True)
    plt.show()
    Ygrid = Xgrid[:,0]*Xgrid[:,1]
    R = np.abs(Ygrid - Yhat)

    plt.scatter(Xgrid[:,0], Xgrid[:,1], c=R)
    plt.show()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(Xgrid[:,0], Xgrid[:,1], R, linewidth=0.2, antialiased=True)
    plt.show()



