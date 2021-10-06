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
            #dZ = np.outer(Y_train - T, V)*(Z>0) #Orginal
            dZ = np.outer(Y_train - T, V)*np.sign(Z)
        else:
            #dZ = (Y_train - T).dot(V.T)*(Z>0) #Orginal
            dZ = (Y_train - T).dot(V.T)*np.sign(Z)
        return X.T.dot(dZ)

    def derivative_b1(self, Z, T, Y_train, V):  #ouput = Y_train, V = weights of 2 layer
        if self.goal == 'regression':
            #dZ = np.outer(Y_train - T, V)*(Z>0) #Orginal
            dZ = np.outer(Y_train - T, V)*np.sign(Z)
        else:
            #dZ = (Y_train - T).dot(V.T)*(Z>0) #Orginal
            dZ = (Y_train - T).dot(V.T)*np.sign(Z)
        return np.sum(dZ, axis = 0)

    def derivative_w2(self, Z, T, Y_train): #Hidden = Z, ouput = Y_train
        return Z.T.dot(Y_train - T)

    def derivative_b2(self, T, Y_train): #ouput = Y_train
        return np.sum(Y_train - T, axis = 0)

class ANN():
    """
    A simple 1 hidden layer Neural Network. Default activation_function is relu (rectifier linear unit),
    can also select tanh, sigmoid. \n
    Can also select between SGD and Batch_SGD gradient descent method (default is Batch_SGD)\n
    Be sure to normalize the data
    """
    def __init__(self, activation_function = "relu", GD_method = "Batch_SGD"):
        self.activation_function = activation_function
        self.GD_method = GD_method
        
    def y2_indicator(self, Y, K, N):
        """
        Creates the indicator matrix for the targets
        """
        ind = np.zeros((N, K))
        ind[np.arange(N), Y[:].astype(np.int32)] = 1
        return ind

    def train_test(self, X, Y):
        K = len(set(Y))
        N = len(Y)

        X_train = X[:]
        Y_train_og = Y[:]
        Y_train = self.y2_indicator(Y_train_og, K, N)
        return X_train, Y_train

    def batch_normal(self, A, eps = 1e-10):
        mu = A.mean(axis=0)
        std = A.std(axis=0)
        np.place(std, std == 0, 1)
        A = (A - mu) / std + eps

        #Y = gam * A + beta
        return A

    def random_weights(self, X, Y, M):
        D = X.shape[1]
        K = len(set(Y))

        W1 = np.random.randn(D, M) / np.sqrt(D)
        b1 = np.zeros(M)
        W2 = np.random.randn(M, K) / np.sqrt(M)
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
            Z[Z < 0] = 0
        else:
            Z = 1 / (1 + np.exp(-X.dot(W1) - b1))
        
        A = Z.dot(W2) + b2
        if self.batch_norm:
            A = self.batch_normal(A)

        if self.goal == 'classification':
            Y_hat = self.softmax(A)
        elif self.goal == 'regression':
            Y_hat = A           
        return Y_hat, Z

    def momentum(self, dW1, db1, dW2, db2, gW1, gb1, gW2, gb2, W1, b1, W2, b2, learning_rate, mu=0.9):
        dW2 = mu * dW2 - learning_rate * gW2
        db2 = mu * db2 - learning_rate * gb2
        dW1 = mu * dW1 - learning_rate * gW1
        db1 = mu * db1 - learning_rate * gb1

        W2 += dW2
        b2 += db2
        W1 += dW1
        b1 += db1

        return W1, b1, W2, b2, dW1, db1, dW2, db2

    def nesterov(self, vW1, vb1, vW2, vb2, gW1, gb1, gW2, gb2, W1, b1, W2, b2, learning_rate, mu=0.9):
        vW2 = mu * vW2 - learning_rate * gW2
        vb2 = mu * vb2 - learning_rate * gb2
        vW1 = mu * vW1 - learning_rate * gW1
        vb1 = mu * vb1 - learning_rate * gb1

        W2 += mu * vW2 - learning_rate * gW2
        b2 += mu * vb2 - learning_rate * gb2
        W1 += mu * vW1 - learning_rate * gW1
        b1 += mu * vb1 - learning_rate * gb1

        return W1, b1, W2, b2, vW1, vb1, vW2, vb2

    def rmsprop(self, cW1, cb1, cW2, cb2, gW1, gb1, gW2, gb2, W1, b1, W2, b2, learning_rate, decay=0.999, eps=1e-10):
        cW2 = decay * cW2 + (1-decay) * gW2 * gW2
        cb2 = decay * cb2 + (1-decay) * gb2 * gb2
        cW1 = decay * cW1 + (1-decay) * gW1 * gW1
        cb1 = decay * cb1 + (1-decay) * gb1 * gb1

        W2 -= learning_rate * gW2 / (np.sqrt(cW2) + eps)
        b2 -= learning_rate * gb2 / (np.sqrt(cb2) + eps)
        W1 -= learning_rate * gW1 / (np.sqrt(cW1) + eps)
        b1 -= learning_rate * gb1 / (np.sqrt(cb1) + eps)

        return W1, b1, W2, b2, cW1, cb1, cW2, cb2

    def adam(self, mW1, mb1, mW2, mb2, aW1, ab1, aW2, ab2, gW1, gb1, gW2, gb2, W1, b1, W2, b2, t, lr, be1=0.99, be2=0.99, eps=1e-10):
        mW1 = be1 * mW1 + (1-be1) * gW1
        mb1 = be1 * mb1 + (1-be1) * gb1
        mW2 = be1 * mW2 + (1-be1) * gW2
        mb2 = be1 * mb2 + (1-be1) * gb2

        aW1 = be2 * aW1 + (1-be2) * gW1 * gW1 
        ab1 = be2 * ab1 + (1-be2) * gb1 * gb1
        aW2 = be2 * aW2 + (1-be2) * gW2 * gW2
        ab2 = be2 * ab2 + (1-be2) * gb2 * gb2

        correction1 = 1 - be1 ** t
        hat_mW1 = mW1 / correction1
        hat_mb1 = mb1 / correction1
        hat_mW2 = mW2 / correction1
        hat_mb2 = mb2 / correction1

        correction2 = 1 - be2 ** t
        hat_aW1 = aW1 / correction2
        hat_ab1 = ab1 / correction2
        hat_aW2 = aW2 / correction2
        hat_ab2 = ab2 / correction2

        t += 1

        W2 -= lr * hat_mW2 / (np.sqrt(hat_aW2) + eps)
        b2 -= lr * hat_mb2 / (np.sqrt(hat_ab2) + eps)
        W1 -= lr * hat_mW1 / (np.sqrt(hat_aW1) + eps)
        b1 -= lr * hat_mb1 / (np.sqrt(hat_ab1) + eps)
        return W1, b1, W2, b2, mW1, mb1, mW2, mb2, aW1, ab1, aW2, ab2, t

    def fit(self, X, Y, epochs=10000, regularization=0, learning_rate=0.001, M=100, goal='classification',
    costs_show=False, batch_size=500, momentum="None", optimizer='None', early_stop=False, batch_norm=True):
        """
        Enter the regularization value (default is 0).\n
        Enter the learning rate (default value is 0.001).\n
        Enter the number of hidden neurons (default value is 5).\n
        Specify if regression or classification (default is multi-class classification).\n
        If costs_show == True, will plot the training costs.\n
        Momentum can be used to speed up training, default = None, simple, nesterov.\n
        Choose between no optimizer, rmsprop or adam.\n
        Early stop, will seace to train the network if training cost < 0.005. \n
        If Batch_SGD is performed the #epochs is #epochs / batch_size + 100
        """
        self.goal = goal
        self.batch_norm = batch_norm
        train_costs = []
        if self.GD_method == 'Batch_SGD':
            num_batches = len(X)//batch_size
            epochs = epochs//batch_size + 100

        if momentum == "simple":
            dW2 = 0
            db2 = 0
            dW1 = 0
            db1 = 0
        elif momentum == 'nesterov':
            vW2 = 0
            vb2 = 0
            vW1 = 0
            vb1 = 0

        if optimizer == "rmsprop":
            cW1 = 1
            cb1 = 1
            cW2 = 1
            cb2 = 1
        elif optimizer == 'adam':
            mW1, mb1, mW2, mb2 = 0, 0, 0, 0
            aW1, ab1, aW2, ab2 = 0, 0, 0, 0
            t = 1

        if goal == 'classification':
            X_train, Y_train = self.train_test(X, Y)
            W1, b1, W2, b2 = self.random_weights(X, Y, M = M)
        else:
            X_train, Y_train = X, Y
            W1, b1, W2, b2 = self.random_weights_reg(X, Y, M = M)

        if self.activation_function == "tanh":
            act_fun = gradient_descent_tan(goal)
        elif self.activation_function == "relu":
            act_fun = gradient_descent_relu(goal)
        else:
            act_fun = gradient_descent_sig(goal)

        for _ in range(epochs):
            if self.GD_method == 'Batch_SGD':
                for j in range(num_batches):
                    Xtrain = X_train[j*batch_size:(j*batch_size + batch_size),]
                    Ytrain = Y_train[j*batch_size:(j*batch_size + batch_size),]

                    pY_train, Z_train = self.forward(Xtrain, W1, b1, W2, b2)

                    if momentum == "simple" or momentum == 'nesterov' or optimizer == 'rmsprop' or optimizer == 'adam':
                        gW2 = act_fun.derivative_w2(Z_train, Ytrain, pY_train) + regularization * W2
                        gb2 = act_fun.derivative_b2(Ytrain, pY_train)  + regularization * b2
                        gW1 = act_fun.derivative_w1(Xtrain, Z_train, Ytrain, pY_train, W2) + regularization * W1
                        gb1 = act_fun.derivative_b1(Z_train, Ytrain, pY_train, W2) + regularization * b1
                        
                        if optimizer == 'rmsprop':
                            W1, b1, W2, b2, cW1, cb1, cW2, cb2 = self.rmsprop(cW1, cb1, cW2, cb2, gW1, gb1, gW2, gb2, W1, b1, W2, b2, learning_rate)
                        elif optimizer == 'adam': 
                            W1, b1, W2, b2, mW1, mb1, mW2, mb2, aW1, ab1, aW2, ab2, t = self.adam(mW1, mb1, mW2, mb2, aW1, ab1, aW2, ab2, 
                            gW1, gb1, gW2, gb2, W1, b1, W2, b2, t, learning_rate)

                        if momentum == 'simple':
                            W1, b1, W2, b2, dW1, db1, dW2, db2 = self.momentum(dW1, db1, dW2, db2, gW1, gb1, gW2, gb2, W1, b1, W2, b2, learning_rate)
                        elif momentum == 'nesterov':
                            W1, b1, W2, b2, vW1, vb1, vW2, vb2 = self.momentum(vW1, vb1, vW2, vb2, gW1, gb1, gW2, gb2, W1, b1, W2, b2, learning_rate)

                    else:
                        W2 -= learning_rate * (act_fun.derivative_w2(Z_train, Ytrain, pY_train) + regularization * W2)
                        b2 -= learning_rate * (act_fun.derivative_b2(Ytrain, pY_train)  + regularization * b2) 
                        W1 -= learning_rate * (act_fun.derivative_w1(Xtrain, Z_train, Ytrain, pY_train, W2) + regularization * W1)
                        b1 -= learning_rate * (act_fun.derivative_b1(Z_train, Ytrain, pY_train, W2) + regularization *b1)

                    if goal == 'classification':
                        ctrain = -np.mean(Ytrain*np.log(pY_train))
                    else:
                        ctrain = -np.mean((pY_train - Ytrain)**2)
                    train_costs.append(ctrain)

            else:
                pY_train, Z_train = self.forward(X_train, W1, b1, W2, b2)
                
                if momentum == "simple" or momentum == 'nesterov' or optimizer == 'rmsprop' or optimizer == 'adam':
                    gW2 = act_fun.derivative_w2(Z_train, Ytrain, pY_train) + regularization * W2
                    gb2 = act_fun.derivative_b2(Ytrain, pY_train)  + regularization * b2
                    gW1 = act_fun.derivative_w1(Xtrain, Z_train, Ytrain, pY_train, W2) + regularization * W1
                    gb1 = act_fun.derivative_b1(Z_train, Ytrain, pY_train, W2) + regularization * b1

                    if optimizer == 'rmsprop':
                        W1, b1, W2, b2, cW1, cb1, cW2, cb2 = self.rmsprop(cW1, cb1, cW2, cb2, gW1, gb1, gW2, gb2, W1, b1, W2, b2, learning_rate)
                    elif optimizer == 'adam':
                        W1, b1, W2, b2, mW1, mb1, mW2, mb2, aW1, ab1, aW2, ab2, t = self.adam(mW1, mb1, mW2, mb2, aW1, ab1, aW2, ab2,
                        gW1, gb1, gW2, gb2, W1, b1, W2, b2, t, learning_rate)

                    if momentum == 'simple':
                        W1, b1, W2, b2, dW1, db1, dW2, db2 = self.momentum(dW1, db1, dW2, db2, gW1, gb1, gW2, gb2, W1, b1, W2, b2, learning_rate, mu=0.9)
                    elif momentum == "nesterov":
                        W1, b1, W2, b2, vW1, vb1, vW2, vb2 = self.momentum(vW1, vb1, vW2, vb2, gW1, gb1, gW2, gb2, W1, b1, W2, b2, learning_rate, mu=0.9)
                else:
                    W2 -= learning_rate * (act_fun.derivative_w2(Z_train, Y_train, pY_train) + regularization * W2)
                    b2 -= learning_rate * (act_fun.derivative_b2(Y_train, pY_train)  + regularization * b2) 
                    W1 -= learning_rate * (act_fun.derivative_w1(X_train, Z_train, Y_train, pY_train, W2) + regularization * W1)
                    b1 -= learning_rate * (act_fun.derivative_b1(Z_train, Y_train, pY_train, W2) + regularization *b1)

                if goal == 'classification':
                    ctrain = -np.mean(Y_train*np.log(pY_train))
                else:
                    ctrain = -np.mean((pY_train - Y_train)**2)
                train_costs.append(ctrain)

            if type(early_stop) == type(0.5):
                if ctrain < early_stop:
                    break
        
        self.W2 = W2
        self.b2 = b2
        self.W1 = W1
        self.b1 = b1

        if costs_show:
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
    
