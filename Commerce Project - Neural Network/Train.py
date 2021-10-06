import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from Pre_proccessing_Data import proccessing_data
from Predict import predict

def y2_indicator(Y, K):
    """
    Creates the indicator matrix for the targets
    """
    N = len(Y)
    ind = np.zeros((N, K))
    ind[np.arange(N), Y[:].astype(np.int32)] = 1
    #print(ind.shape)
    #for i in range(N):
    #    print(Y[i], ind[i])
    return ind

def train_test():
    X, Y = prd.get_data(filename = "ecommerce_data.csv")
    X, Y = shuffle(X, Y)
    K = len(set(Y))

    X_train = X[:-100]
    Y_train = Y[:-100]
    Y_train_ind = y2_indicator(Y_train, K)

    X_test = X[-100:]
    Y_test = Y[-100:]
    Y_test_ind = y2_indicator(Y_test, K)

    return X, Y, X_train, Y_train, X_test, Y_test, Y_train_ind, Y_test_ind

def cross_entropy(T, pY):
    return -np.mean(T*np.log(pY))

class gradient_ascent_sig():
    """
    W2 += learning_rate * ga.derivative_w2(hidden, T, output)
    b2 += learning_rate * ga.derivative_b2(T, output)
    W1 += learning_rate * ga.derivative_w1(X, hidden, T, output, W2)
    b1 += learning_rate * ga.derivative_b1(hidden, T, output, W2)
    """
    def derivative_w2(self, Z, T, Y_hat): #Hidden = Z, ouput = Y_hat
        return Z.T.dot(T - Y_hat)

    def derivative_b2(self, T, Y_hat): #ouput = Y_hat
        return np.sum(T - Y_hat, axis = 0)

    def derivative_w1(self, X, Z, T, Y_hat, V):   #Hidden = Z, ouput = Y_hat, V = weights of 2 layer
        deltaZ = (T-Y_hat).dot(V.T)*Z*(1-Z)
        return X.T.dot(deltaZ)

    def derivative_b1(self, Z, T, Y_hat, V):  #ouput = Y_hat, V = weights of 2 layer
        return np.sum((T - Y_hat).dot(V.T) * Z * (1 - Z), axis = 0)

class gradient_descent_sig():
    """
    W2 -= learning_rate * ga.derivative_w2(hidden, T, output)
    b2 -= learning_rate * ga.derivative_b2(T, output)
    W1 -= learning_rate * ga.derivative_w1(X, hidden, T, output, W2)
    b1 -= learning_rate * ga.derivative_b1(hidden, T, output, W2)
    """
    def derivative_w1(self, X, Z, T, Y_hat, V):   #Hidden = Z, ouput = Y_hat, V = weights of 2 layer
        return X.T.dot((Y_hat - T).dot(V.T)*Z*(1-Z))

    def derivative_b1(self, Z, T, Y_hat, V):  #ouput = Y_hat, V = weights of 2 layer
        return np.sum((Y_hat - T).dot(V.T) * Z * (1 - Z), axis = 0)

    def derivative_w2(self, Z, T, Y_hat): #Hidden = Z, ouput = Y_hat
        return Z.T.dot(Y_hat - T)

    def derivative_b2(self, T, Y_hat): #ouput = Y_hat
        return np.sum(Y_hat - T, axis = 0)

class gradient_descent_tan():
    """
    W2 -= learning_rate * ga.derivative_w2(hidden, T, output)
    b2 -= learning_rate * ga.derivative_b2(T, output)
    W1 -= learning_rate * ga.derivative_w1(X, hidden, T, output, W2)
    b1 -= learning_rate * ga.derivative_b1(hidden, T, output, W2)
    """
    def derivative_w1(self, X, Z, T, Y_hat, V):   #Hidden = Z, ouput = Y_hat, V = weights of 2 layer
        return X.T.dot((Y_hat - T).dot(V.T)*(1-Z*Z))

    def derivative_b1(self, Z, T, Y_hat, V):  #ouput = Y_hat, V = weights of 2 layer
        return np.sum((Y_hat - T).dot(V.T) * (1 - Z*Z), axis = 0)

    def derivative_w2(self, Z, T, Y_hat): #Hidden = Z, ouput = Y_hat
        return Z.T.dot(Y_hat - T)

    def derivative_b2(self, T, Y_hat): #ouput = Y_hat
        return np.sum(Y_hat - T, axis = 0)

class gradient_ascent_tan():
    """
    W2 += learning_rate * ga.derivative_w2(hidden, T, output)
    b2 += learning_rate * ga.derivative_b2(T, output)
    W1 += learning_rate * ga.derivative_w1(X, hidden, T, output, W2)
    b1 += learning_rate * ga.derivative_b1(hidden, T, output, W2)
    """
    def derivative_w2(self, Z, T, Y_hat): #Hidden = Z, ouput = Y_hat
        return Z.T.dot(T - Y_hat)

    def derivative_b2(self, T, Y_hat): #ouput = Y_hat
        return np.sum(T - Y_hat, axis = 0)

    def derivative_w1(self, X, Z, T, Y_hat, V):   #Hidden = Z, ouput = Y_hat, V = weights of 2 layer
        deltaZ = (T-Y_hat).dot(V.T)*(1-Z**2)
        return X.T.dot(deltaZ)

    def derivative_b1(self, Z, T, Y_hat, V):  #ouput = Y_hat, V = weights of 2 layer
        return np.sum((T - Y_hat).dot(V.T) * (1 - Z**2), axis = 0)

def backprop(learning_rate = 0.001, M = 5):
    X, Y, X_train, Y_train, X_test, Y_test, Y_train_ind, Y_test_ind = train_test()

    #Randomise the weights:
    W1, b1, W2, b2 = pd.random_weights(X, Y, M = M)

    train_costs = []
    test_costs = []

    for i in range(10000):
        pY_train, Z_train = pd.forward(X_train, W1, b1, W2, b2)
        pY_test, _ = pd.forward(X_test, W1, b1, W2, b2)

        ctrain = cross_entropy(Y_train_ind, pY_train)
        ctest = cross_entropy(Y_test_ind, pY_test)
        if i % 1000 == 0:
            print(i, ctrain, ctest)
        train_costs.append(ctrain)
        test_costs.append(ctest)

        #Gradient Descent
        W2 -= learning_rate * gdt.derivative_w2(Z_train, Y_train_ind, pY_train)
        b2 -= learning_rate * gdt.derivative_b2(Y_train_ind, pY_train)
        W1 -= learning_rate * gdt.derivative_w1(X_train, Z_train, Y_train_ind, pY_train, W2)
        b1 -= learning_rate * gdt.derivative_b1(Z_train, Y_train_ind, pY_train, W2)
    
    class_rate_train = pd.classification_rate(Y_train, pY_train)
    class_rate_test = pd.classification_rate(Y_test, pY_test)

    print("Final train classification_rate: ", class_rate_train)
    print("Final test classification_rate: ", class_rate_test)

    return train_costs, test_costs

def plot_graph(train_costs, test_costs):
    legend1, = plt.plot(train_costs, label = 'train cost')
    legend2, = plt.plot(test_costs, label = 'test costs')
    plt.legend([legend1, legend2])
    plt.show()

if __name__ == "__main__":
    pd = predict()
    prd = proccessing_data()
    gdt = gradient_descent_tan()
    gds = gradient_descent_sig()

    train_costs, test_costs = backprop()
    plot_graph(train_costs, test_costs)


    