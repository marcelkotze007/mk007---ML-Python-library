import numpy as np
import matplotlib.pyplot as plt

class gradient_descent_sig():
    """
    W2 -= learning_rate * ga.derivative_w2(hidden, T, output)
    b2 -= learning_rate * ga.derivative_b2(T, output)
    W1 -= learning_rate * ga.derivative_w1(X, hidden, T, output, W2)
    b1 -= learning_rate * ga.derivative_b1(hidden, T, output, W2)
    """
    def __init__(self, goal):
        self.goal = goal

    def derivative_w1(self, X, Z, T, Y_train, V):   #Hidden = Z, ouput = Y_train, V = weights of 2 layer
        if self.goal == 'regression':
            dZ = np.outer(Y_train - T, V)*Z*(1-Z)
        else:
            dZ = (Y_train - T).dot(V.T)*Z*(1-Z)
        return X.T.dot(dZ)

    def derivative_b1(self, Z, T, Y_train, V):  #ouput = Y_train, V = weights of 2 layer
        if self.goal == 'regression':
            dZ = np.outer(Y_train - T, V)*Z*(1-Z)
        else:
            dZ = (Y_train - T).dot(V.T)*Z*(1-Z)
        return np.sum(dZ, axis = 0)

    def derivative_w2(self, Z, T, Y_train): #Hidden = Z, ouput = Y_train
        return Z.T.dot(Y_train - T)

    def derivative_b2(self, T, Y_train): #ouput = Y_train
        return np.sum(Y_train - T, axis = 0)

class gradient_descent_tan():
    """
    W2 -= learning_rate * ga.derivative_w2(hidden, T, output)
    b2 -= learning_rate * ga.derivative_b2(T, output)
    W1 -= learning_rate * ga.derivative_w1(X, hidden, T, output, W2)
    b1 -= learning_rate * ga.derivative_b1(hidden, T, output, W2)
    """
    def __init__(self, goal):
        self.goal = goal

    def derivative_w1(self, X, Z, T, Y_train, V):   #Hidden = Z, ouput = Y_train, V = weights of 2 layer
        if self.goal == 'regression':
            dZ = np.outer(Y_train - T, V)*(1-Z*Z)
        else:
            dZ = (Y_train - T).dot(V.T)*(1-Z*Z)
        return X.T.dot(dZ)

    def derivative_b1(self, Z, T, Y_train, V):  #ouput = Y_train, V = weights of 2 layer
        if self.goal == 'regression':
            dZ = np.outer(Y_train - T, V)*(1-Z*Z)
        else:
            dZ = (Y_train - T).dot(V.T)*(1-Z*Z)
        return np.sum(dZ, axis = 0)

    def derivative_w2(self, Z, T, Y_train): #Hidden = Z, ouput = Y_train
        return Z.T.dot(Y_train - T)

    def derivative_b2(self, T, Y_train): #ouput = Y_train
        return np.sum(Y_train - T, axis = 0)

class gradient_descent_relu():
    """
    W2 -= learning_rate * ga.derivative_w2(hidden, T, output)
    b2 -= learning_rate * ga.derivative_b2(T, output)
    W1 -= learning_rate * ga.derivative_w1(X, hidden, T, output, W2)
    b1 -= learning_rate * ga.derivative_b1(hidden, T, output, W2)
    """
    def __init__(self, goal):
        self.goal = goal

    def derivative_w1(self, X, Z, T, Y_train, V):   #Hidden = Z, ouput = Y_train, V = weights of 2 layer
        if self.goal == 'regression':
            dZ = np.outer(Y_train - T, V)*(Z>0)
        else:
            dZ = (Y_train - T).dot(V.T)*(Z>0)
        return X.T.dot(dZ)

    def derivative_b1(self, Z, T, Y_train, V):  #ouput = Y_train, V = weights of 2 layer
        if self.goal == 'regression':
            dZ = np.outer(Y_train - T, V)*(Z>0)
        else:
            dZ = (Y_train - T).dot(V.T)*(Z>0)
        return np.sum(dZ, axis = 0)

    def derivative_w2(self, Z, T, Y_train): #Hidden = Z, ouput = Y_train
        return Z.T.dot(Y_train - T)

    def derivative_b2(self, T, Y_train): #ouput = Y_train
        return np.sum(Y_train - T, axis = 0)

class ANN():
    """
    A simple 1 hidden layer Neural Network. Default activation_function is relu, can also select tanh, sigmoid
    """
    def __init__(self, activation_function = "relu"):
        self.activation_function = activation_function
        
    def y2_indicator(self, Y, K, N):
        """
        Creates the indicator matrix for the targets
        """
        ind = np.zeros((N, K))
        ind[np.arange(N), Y[:].astype(np.int32)] = 1
        #print(ind.shape)
        #for i in range(N):
        #    print(Y[i], ind[i])
        return ind

    def train_test(self, X, Y):
        K = len(set(Y))
        N = len(Y)

        #X_train = X[:-(N//5)]
        X_train = X[:]
        #Y_train_og = Y[:-(N//5)]
        Y_train_og = Y[:]
        Y_train = self.y2_indicator(Y_train_og, K, N)

        #X_test = X[-(N//5):]
        #Y_test = Y[-(N//5):]
        #Y_test_ind = self.y2_indicator(Y_test, K, N)

        #return X_train, Y_train_og, X_test, Y_test_og, Y_train, Y_test
        return X_train, Y_train

    def random_weights(self, X, Y, M):
        D = X.shape[1]
        K = len(set(Y))

        W1 = np.random.randn(D, M)
        b1 = np.zeros(M)
        W2 = np.random.randn(M, K)
        b2 = np.zeros(K)

        return W1, b1, W2, b2

    def random_weights_reg(self, X, Y, M):
        D = X.shape[1]

        W1 = np.random.randn(D, M) / np.sqrt(D)
        b1 = np.zeros(M)
        W2 = np.random.randn(M) / np.sqrt(M)
        b2 = 0

        return W1, b1, W2, b2

    def softmax(self, a):
        """
        calculates the prob 
        """
        expA = np.exp(a)
        return expA / expA.sum(axis=1, keepdims = True)

    def forward(self, X, W1, b1, W2, b2):
        if self.activation_function == "tanh":
            Z = np.tanh(X.dot(W1) + b1)
        elif self.activation_function == "relu":
            Z = X.dot(W1) + b1
            Z = Z * (Z > 0)
        else:
            Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
        A = Z.dot(W2) + b2
        if self.goal == 'classification':
            Y_hat = self.softmax(A)
        elif self.goal == 'regression':
            Y_hat = A
            
        return Y_hat, Z

    def fit(self, X, Y, epochs = 10000, regularization = 0, learning_rate = 0.001, M = 5, goal = 'classification',
    costs = False):
        """
        Enter the regularization value (default is 0).\n
        Enter the learning rate (default value is 0.001).\n
        Enter the number of hidden neurons (default value is 5).\n
        Specify if regression or classification (default is multi-class classification)\n
        If Costs == True, will plot the training costs
        """
        self.goal = goal
        train_costs = []
        if goal == 'classification':
            X_train, Y_train = self.train_test(X, Y)
            W1, b1, W2, b2 = self.random_weights(X, Y, M = M)
        else:
            X_train, Y_train = X, Y
            W1, b1, W2, b2 = self.random_weights_reg(X, Y, M = M)

        if self.activation_function == "tanh":
            gdt = gradient_descent_tan(goal)
            for i in range(epochs):
                pY_train, Z_train = self.forward(X_train, W1, b1, W2, b2)
                
                W2 -= learning_rate * (gdt.derivative_w2(Z_train, Y_train, pY_train) + regularization * W2)
                b2 -= learning_rate * (gdt.derivative_b2(Y_train, pY_train)  + regularization * b2) 
                W1 -= learning_rate * (gdt.derivative_w1(X_train, Z_train, Y_train, pY_train, W2) + regularization * W1)
                b1 -= learning_rate * (gdt.derivative_b1(Z_train, Y_train, pY_train, W2) + regularization *b1)

                if costs:
                    if goal == 'classification':
                        ctrain = -np.mean(Y_train*np.log(pY_train))
                    else:
                        ctrain = -np.mean((pY_train - Y_train)**2)
                    train_costs.append(ctrain)

        elif self.activation_function == "relu":
            gdr = gradient_descent_relu(goal)
            for i in range(epochs):
                pY_train, Z_train = self.forward(X_train, W1, b1, W2, b2)
               
                W2 -= learning_rate * (gdr.derivative_w2(Z_train, Y_train, pY_train) + regularization * W2)
                b2 -= learning_rate * (gdr.derivative_b2(Y_train, pY_train) + regularization * b2)
                W1 -= learning_rate * (gdr.derivative_w1(X_train, Z_train, Y_train, pY_train, W2) + regularization * W1)
                b1 -= learning_rate * (gdr.derivative_b1(Z_train, Y_train, pY_train, W2) + regularization *b1)
                
                if costs:
                    if goal == 'classification':
                        ctrain = -np.mean(Y_train*np.log(pY_train))
                    else:
                        ctrain = -np.mean((pY_train - Y_train)**2)
                    train_costs.append(ctrain)

        else:
            gds = gradient_descent_sig(goal)
            for i in range(epochs):
                pY_train, Z_train = self.forward(X_train, W1, b1, W2, b2)
                
                W2 -= learning_rate * (gds.derivative_w2(Z_train, Y_train, pY_train) + regularization * W2)
                b2 -= learning_rate * (gds.derivative_b2(Y_train, pY_train) + regularization * b2)
                W1 -= learning_rate * (gds.derivative_w1(X_train, Z_train, Y_train, pY_train, W2) + regularization * W1)
                b1 -= learning_rate * (gds.derivative_b1(Z_train, Y_train, pY_train, W2) + regularization *b1)

                if costs:
                    if goal == 'classification':
                        ctrain = -np.mean(Y_train*np.log(pY_train))
                    else:
                        ctrain = -np.mean((pY_train - Y_train)**2)
                    train_costs.append(ctrain)

        self.W2 = W2
        self.b2 = b2
        self.W1 = W1
        self.b1 = b1

        if costs:
            plt.plot(train_costs)
            plt.title("Training Costs")
            plt.show()

    def predict(self, X_test):
        pY_test, _ = self.forward(X_test, self.W1, self.b1, self.W2, self.b2)
        return pY_test
    
    def score(self, X_test, Y):
        pY_test = self.predict(X_test)
        if self.goal == 'classification':
            P = np.argmax(pY_test, axis=1)
        elif self.goal == 'regression':
            P = pY_test
        return np.mean(Y == P)

if __name__ == "__main__":
    ANN = ANN()
    
