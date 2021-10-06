import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import  shuffle
from Pre_proccessing_Data import proccessing_data
from Logistic_Predict import logistic_predict

LP = logistic_predict()
PPD = proccessing_data()

filename = "C:/Users/Marcel/OneDrive/Python Courses/Deep Learning/Commerce Project/ecommerce_data.csv"

class logistic_train:
    def create_train_and_test(self, filename):
        X, Y = PPD.get_data(filename)

        #Split data into train and test sets:
        X_train = X[:-100]
        Y_train = Y[:-100]
        X_test = X[-100:]
        Y_test = Y[-100:]

        #Return the data from the first to classes (binary classes)
        X2_train = X_train[Y_train <= 1]
        Y2_train = Y_train[Y_train <= 1]
        X2_test = X_test[Y_test <= 1]
        Y2_test = Y_test[Y_test <= 1]

        return X2_train, Y2_train, X2_test, Y2_test 
    
    def cross_entropy_error(self, Y, Y_hat):
        return -np.mean(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))
    
    def gradient_decent(self, learning_rate = 0.01):
        X2_train, Y2_train, X2_test, Y2_test = LT.create_train_and_test(filename)

        #Randomly initialize weights:
        D = X2_train.shape[1]
        w = np.random.randn(D)
        b = 0

        train_costs = []
        test_costs = []

        for i in range(10000):
            P_Y_train = LP.forward(X2_train, w, b)
            P_Y_test = LP.forward(X2_test, w, b)

            cost_train = LT.cross_entropy_error(Y2_train, P_Y_train)
            cost_test = LT.cross_entropy_error(Y2_test, P_Y_test)

            train_costs.append(cost_train)
            test_costs.append(cost_test)

            #Gradient descent:
            w -= learning_rate * X2_train.T.dot(P_Y_train - Y2_train)
            b -= learning_rate * (P_Y_train - Y2_train).sum()

            if i % 1000 == 0:
                print(i, cost_train, cost_test)
        
        class_rate_train = LP.classification_rate(Y2_train, np.round(P_Y_train))
        class_rate_test = LP.classification_rate(Y2_test, np.round(P_Y_test))

        print("Final train classification_rate: ", class_rate_train)
        print("Final test classification_rate: ", class_rate_test)

        return train_costs, test_costs
    
    def plot_graph(self, train_costs, test_costs):
        legend1, = plt.plot(train_costs, label = 'train cost')
        legend2, = plt.plot(test_costs, label = 'test costs')
        plt.legend([legend1, legend2])
        plt.show()

LT = logistic_train()

train_costs, test_costs = LT.gradient_decent()
LT.plot_graph(train_costs, test_costs)