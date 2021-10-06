import numpy as np
import matplotlib.pyplot as plt

def create():
    # create the data
    Nclass = 500
    D = 2 # dimensionality of input
    #M = 3 # hidden layer size
    K = 3 # number of classes

    X1 = np.random.randn(Nclass, D) + np.array([0, -2])
    X2 = np.random.randn(Nclass, D) + np.array([2, 2])
    X3 = np.random.randn(Nclass, D) + np.array([-2, 2])
    X = np.vstack([X1, X2, X3])

    Y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    N = len(Y)
    # turn Y into an indicator matrix for training
    T = np.zeros((N, K))
    T[np.arange(N), Y[:].astype(np.int32)] = 1

    # let's see what it looks like
    plt.scatter(X[:,0], X[:,1], c=Y, s=100, alpha=0.5)
    plt.show()
    return X, T, Y

def classification_rate(Y, P):
    return np.mean(Y==P)

def cost(T, Y):
    tot = T * np.log(Y)
    return tot.sum()

def derivative_w3(Z, T, Y):
    ret6 = Z.T.dot(T - Y)
    return ret6

def derivative_w2(X, Z, T, Y, W3):

def derivative_w1(X, Z, T, Y, W2):
    dZ = (T - Y).dot(W2.T) * Z * (1 - Z)
    ret2 = X.T.dot(dZ)
    return ret2

def derivative_b3(T, Y):
    return (T - Y).sum(axis=0)

def derivative_b2()

def derivative_b1(T, Y, W2, Z):
    return ((T - Y).dot(W2.T) * Z * (1 - Z)).sum(axis=0)

def forward(X, W1, b1, W2, b2, W3, b3):
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    A = Z.dot(W2) + b2
    expA = np.exp(A)
    Y = expA / expA.sum(axis=1, keepdims=True)
    return Y, Z

def main():
    D = 2 # dimensionality of input
    M = 3 # hidden layer size
    M1 = 3
    K = 3 # number of classes

    X, T, Y = create()
    W1 = np.random.randn(D, M) / np.sqrt(D)
    b1 = np.random.randn(M)
    W2 = np.random.randn(M, M1) / np.sqrt(M)
    b2 = np.random.randn(M1)
    W3 = np.random.randn(M1, K) / np.sqrt(M1)
    b3 = np.random.randn(K)

    learning_rate = 1e-3
    costs = []
    for epoch in range(1000):
        output, hidden = forward(X, W1, b1, W2, b2, W3, b3)
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis=1)
            r = classification_rate(Y, P)
            print("cost:", c, "classification_rate:", r)
            costs.append(c)

        W3 += learning_rate * derivative_w3(hidden, T, output)
        b3 += learning_rate * derivative_b3(T, output)
        W2 += learning_rate * derivative_w2(hidden, T, output)
        b2 += learning_rate * derivative_b2(T, output)
        W1 += learning_rate * derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate * derivative_b1(T, output, W2, hidden)

    plt.plot(costs)
    plt.show()    



if __name__ == "__main__":
    main()
