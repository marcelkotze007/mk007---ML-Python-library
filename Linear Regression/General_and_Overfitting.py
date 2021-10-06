import numpy as np
import matplotlib.pyplot as plt

#Create random data and plot it

class Overfitting:
    def get_data(self):
        N = 100
        X = np.linspace(0, 6*np.pi, N)
        Y = np.sin(X)

        return X, Y
    
    def plot_data(self, X, Y):
        plt.plot(X, Y)
        plt.show()

    def make_poly(self, X, deg): #Creates a polynomial linear regression data
        n = len(X)    #Returns the amount of items in a container
        data = [np.ones(n)]
        for d in range(deg):
            data.append(X**(d+1))
        
        return np.vstack(data).T
    
    def fit(self, X, Y):
        #Finds the w for inputs and outputs
        return np.linalg.solve(X.T.dot(X), X.T.dot(Y))

    def fit_and_display(self, X, Y, sample, deg):
        """
        The sample states how many samples to take from x and y to form a training set
        """
        N = len(X)
        train_idx = np.random.choice(N, sample)
        Xtrain = X[train_idx]
        Ytrain = Y[train_idx]

        plt.scatter(Xtrain, Ytrain)
        plt.show()

        #Fit Polynomial:
        Xtrain_poly = Of.make_poly(Xtrain, deg)
        w = Of.fit(Xtrain_poly, Ytrain)

        #Display the Polynomial:
        X_poly = Of.make_poly(X, deg)
        Y_hat = X_poly.dot(w)
        
        plt.plot(X, Y)
        plt.plot(X, Y_hat)
        plt.scatter(Xtrain, Ytrain)
        plt.title("Deg = %d" %deg)
        plt.show()

        return Y_hat
    
    def loop(self, X, Y):
        for deg in (5, 6, 7, 8, 9):
            Of.fit_and_display(X, Y, 10, deg)

        Of.plot_train_vs_test_curves(X, Y)

    def get_mse(self, Y, Y_hat):
        """
        Calculates the mean_squared_error
        """
        d = Y - Y_hat
        return d.dot(d) / len(d)
    
    def plot_train_vs_test_curves(self, X, Y, sample = 20, max_deg = 20):
        N = len(X)
        train_idx = np.random.choice(N, sample)
        Xtrain = X[train_idx]
        Ytrain = Y[train_idx]

        test_idx = [idx for idx in range(N) if idx not in train_idx]
        #test_idx = np.random.choice(N, sample)
        Xtest = X[test_idx]
        Ytest = Y[test_idx]

        mse_trains = []
        mse_tests = []
        for deg in range(max_deg + 1):
            Xtrain_poly = Of.make_poly(Xtrain, deg)
            w = Of.fit(Xtrain_poly, Ytrain)
            Yhat_train = Xtrain_poly.dot(w)
            mse_train = Of.get_mse(Ytrain, Yhat_train)

            Xtest_poly = Of.make_poly(Xtest, deg)
            Yhat_test = Xtest_poly.dot(w)
            mse_test = Of.get_mse(Ytest, Yhat_test)

            mse_trains.append(mse_train)
            mse_tests.append(mse_test)
        
        plt.plot(mse_trains, label = "Train mse")
        plt.plot(mse_tests, label = "Test mse")
        plt.legend()
        plt.show()

        plt.plot(mse_trains, label = "Train mse")
        plt.legend()
        plt.show()


Of = Overfitting()
X, Y = Of.get_data()
#Of.plot_data(X, Y)
Of.loop(X, Y)
