import numpy as np 
import matplotlib.pyplot as plt

def generate_Guassian_cloud(N = 500):
    """
    Generates 3 Guassian clouds
    """
    #first cloud is centred at (0, -2)
    X1 = np.random.randn(500, 2) + np.array([0, -2])
    #second cloud is centred at (2, 2)
    X2 = np.random.randn(500, 2) + np.array([2, 2])
    #Third cloud is centred at (-2, 2)
    X3 = np.random.randn(500, 2) + np.array([-2, 2])
    X = np.vstack((X1,X2,X3))
    Y = np.array([0]*N + [1]*N + [2]*N)

    plt.scatter(X[:, 0], X[:,1], c = Y, s=100, alpha=0.5)
    plt.show()
    return X, Y

#Number of features
D = 2
#Number of neurons in the hidden layer (hidden layer size)
M = 3
#NUmber of classes
K = 3

#initilize the weights and baises for the input layer to the hidden layer:
W1 = np.random.randn(D, M)
b1 = np.random.randn(M)
#initilize the weights and biases for the hidden layer to the output layer
W2 = np.random.randn(M, K)
b2 = np.random.randn(K)

def forward(X, W1, b1, W2, b2):
    #use the sigmoid nonlinearity in the hidden layer
    # Z is the value at the hidden layer
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))

    #Calculate the Softmax of the next layer
    A = Z.dot(W2) + b2
    exp_A = np.exp(A)
    #calculate the prob of each element in each row, with each row.sum(axis = 1) to 1
    Y_hat = exp_A / exp_A.sum(axis = 1, keepdims = True)

    return Y_hat

def classification_rate(Y, Y_hat):
    P = np.argmax(Y_hat, axis=1)
    return np.mean(Y == P)

if __name__ == "__main__":
    X, Y = generate_Guassian_cloud()
    Y_hat = forward(X, W1, b1, W2, b2)
    if len(Y_hat) == len(Y):
        print("len(P) == len(Y)")
    print("Classification rate for randomly choses weights:", classification_rate(Y, Y_hat))
