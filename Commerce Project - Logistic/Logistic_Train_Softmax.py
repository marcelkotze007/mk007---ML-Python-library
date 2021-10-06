import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

from Pre_proccessing_Data import proccessing_data

class prediction():
    def softmax(self, a):
        exp_a = np.exp(a)
        z = exp_a / exp_a.sum(axis = 1, keepdims = True)
        return z

    def forward(self, X, W, b):
        Y_hat = self.softmax(X.dot(W) + b)
        return Y_hat
    
    def predict(self, Y_hat):
        return np.argmax(Y_hat, axis = 1)  #P_Y_Given_X = Y_hat

    def classification_rate(self, Y, P):
        return np.mean(Y == P)

class train():
    def y2_indicator(self, Y, K):
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

    def train_test(self):
        X, Y = prd.get_data(filename = "ecommerce_data.csv")
        X, Y = shuffle(X, Y)
        K = len(set(Y))

        X_train = X[:-100]
        Y_train = Y[:-100]
        Y_train_ind = self.y2_indicator(Y_train, K)

        X_test = X[-100:]
        Y_test = Y[-100:]
        Y_test_ind = self.y2_indicator(Y_test, K)

        return X, Y, X_train, Y_train, X_test, Y_test, Y_train_ind, Y_test_ind
  
    def cross_entropy(self, T, pY):
        return -np.mean(T*np.log(pY))

    def gradient_descent(self, learning_rate = 0.005):
        X, Y, X_train, Y_train, X_test, Y_test, Y_train_ind, Y_test_ind = self.train_test()

        D = X.shape[1]
        K = len(set(Y))

        #Randomise the weights:
        W = np.random.randn(D, K)
        b = np.zeros(K)

        train_costs = []
        test_costs = []

        for i in range(10000):
            pY_train = pd.forward(X_train, W, b)
            pY_test = pd.forward(X_test, W, b)

            ctrain = self.cross_entropy(Y_train_ind, pY_train)
            ctest = self.cross_entropy(Y_test_ind, pY_test)
            if i % 1000 == 0:
                print(i, ctrain, ctest)
            train_costs.append(ctrain)
            test_costs.append(ctest)

            #Gradient Descent
            W -= learning_rate * X_train.T.dot(pY_train - Y_train_ind)
            b -= learning_rate * (pY_train - Y_train_ind).sum(axis = 0)
        
        class_rate_train = pd.classification_rate(Y_train, pd.predict(pY_train))
        class_rate_test = pd.classification_rate(Y_test, pd.predict(pY_test))

        print("Final train classification_rate: ", class_rate_train)
        print("Final test classification_rate: ", class_rate_test)

        return train_costs, test_costs

    def plot_graph(self, train_costs, test_costs):
        legend1, = plt.plot(train_costs, label = 'train cost')
        legend2, = plt.plot(test_costs, label = 'test costs')
        plt.legend([legend1, legend2])
        plt.show()

if __name__ == "__main__":
    prd = proccessing_data()
    pd = prediction()
    tr = train()

    train_costs, test_costs = tr.gradient_descent(learning_rate=0.001)
    tr.plot_graph(train_costs, test_costs)

    
    