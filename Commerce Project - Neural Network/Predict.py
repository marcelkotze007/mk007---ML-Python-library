import numpy as np 
from Pre_proccessing_Data import proccessing_data

class predict():
    def random_weights(self, X, Y, M = 5):
        D = X.shape[1]
        K = len(set(Y))
        
        W1 = np.random.randn(D, M)
        b1 = np.zeros(M)
        W2 = np.random.randn(M, K)
        b2 = np.zeros(K)

        return W1, b1, W2, b2

    def softmax(self, a):
        """
        calculates the prob 
        """
        expA = np.exp(a)
        return expA / expA.sum(axis=1, keepdims = True)

    def forward(self, X, W1, b1, W2, b2):
        #Z = 1 / (1 + np.exp(-X.dot(W1) - b1)) #One possible formula to use
        Z = np.tanh(X.dot(W1) + b1)
        A = Z.dot(W2) + b2
        Y_hat = self.softmax(A)
        return Y_hat, Z

    def classification_rate(self, Y, Y_hat):
        #Selects the max value and returns its position
        P = np.argmax(Y_hat, axis = 1)
        return np.mean(Y == P)

if __name__ == "__main__":
    pd = predict()
