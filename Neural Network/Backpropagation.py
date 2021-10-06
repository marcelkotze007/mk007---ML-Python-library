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

def random_weights(D = 2, M = 3, K = 3):
    #Number of features
    #D = 2
    #Number of neurons in the hidden layer (hidden layer size)
    #M = 3
    #Number of classes
    #K = 3

    #initilize the weights and baises for the input layer to the hidden layer:
    W1 = np.random.randn(D, M)
    b1 = np.random.randn(M)
    #initilize the weights and biases for the hidden layer to the output layer
    W2 = np.random.randn(M, K)
    b2 = np.random.randn(K)

    return W1, b1, W2, b2

def forward(X, W1, b1, W2, b2):
    #use the sigmoid nonlinearity in the hidden layer
    # Z is the value at the hidden layer
    Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
    #Calculate the Softmax of the next layer
    A = Z.dot(W2) + b2
    exp_A = np.exp(A)
    #calculate the prob of each element in each row, with each row.sum(axis = 1) to 1
    Y_hat = exp_A / exp_A.sum(axis = 1, keepdims = True)

    return Y_hat, Z #Returns the hidden layer as well as Z is required to calculate the gradient

def cost(T, Y_hat):
    tot = T * np.log(Y_hat) #output = Y_hat
    return tot.sum()

def classification_rate(Y, P):
    return np.mean(Y == P)

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
        return X.T.dot((Y_hat - T).dot(V.T)*(1-Z**2))

    def derivative_b1(self, Z, T, Y_hat, V):  #ouput = Y_hat, V = weights of 2 layer
        return np.sum((Y_hat - T).dot(V.T) * (1 - Z**2), axis = 0)

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

def backprop(X, T, Y, learning_rate = 10e-7):
    costs = []
    W1, b1, W2, b2 = random_weights()
    for epoch in range(100000):
        output, hidden = forward(X, W1, b1, W2, b2) #output = Y_hat
        if epoch % 100 == 0:
            c = cost(T, output)
            P = np.argmax(output, axis = 1)
            class_rate = classification_rate(Y, P)
            print('cost: %s  classification rate: %s' %(c, class_rate))
            costs.append(c)
        
        #Gradient Ascent: Backwards of gradient decend, thus a + and the t-y instead of y-t
        W2 += learning_rate * gas.derivative_w2(hidden, T, output)
        b2 += learning_rate * gas.derivative_b2(T, output)
        W1 += learning_rate * gas.derivative_w1(X, hidden, T, output, W2)
        b1 += learning_rate * gas.derivative_b1(hidden, T, output, W2)
    
    plt.plot(costs)
    plt.title("The cost curve")
    plt.show()

def main():
    X, Y = generate_Guassian_cloud()

    N = len(Y)
    K = 3
    #Turn the targets into an indicator variable - One-hot encoding
    T = np.zeros((N, K))
    #for i in range(N):
    #    T[i, Y[i]] = 1
    T[np.arange(N), Y[:].astype(np.int32)] = 1

    backprop(X, T, Y)

if __name__ == "__main__":
    gas = gradient_ascent_sig()
    #gds = gradient_descent_sig()
    #gdt = gradient_descent_tan()
    #gat = gradient_ascent_tan()
    main()