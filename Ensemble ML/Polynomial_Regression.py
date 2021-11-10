import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse

NUM_DATASETS = 50
NOISE_VARIANCE = 0.5
MAX_POLY = 12
N = 25
Ntrain = int(0.9*N)

np.random.seed(2)

def Make_Poly(x, D):
    """
    Formats the data
    """
    N = len(x)
    X = np.empty((N,D+1))
    for d in range(D+1):
        X[:,d] = x**d
        if d > 1:
            # Normalize the data
            X[:,d] = (X[:,d] - X[:,d].mean()) / X[:,d].std()

    return X

def Generate_Data_Graph():
    x = np.linspace(-np.pi, np.pi , 100)
    y = np.sin(x)

    return x, y

def Generate_Data_Points():
    X = np.linspace(-np.pi, np.pi, N)
    X = np.random.shuffle(X)
    f_X = np.sin(X)

    return X, f_X

def Create_Empty_Datasets():
    train_scores = np.zeros((NUM_DATASETS, MAX_POLY))
    test_scores = np.zeros((NUM_DATASETS, MAX_POLY))
    train_predictions = np.zeros((Ntrain, NUM_DATASETS, MAX_POLY))
    prediction_curves = np.zeros((100, NUM_DATASETS, MAX_POLY))

    return train_scores, test_scores, train_predictions, prediction_curves

if __name__ == '__main__':
    x_axis, y_axis = Generate_Data_Graph() 
    X, f_X = Generate_Data_Points()

    X_poly = Make_Poly(X, MAX_POLY)

    train_scores, test_scores, train_predictions, prediction_curves = Create_Empty_Datasets()

    model = LinearRegression

    for k in range(NUM_DATASETS):
        Y = f_X + np.random.randn(N)*NOISE_VARIANCE

        Xtrain = X_poly[:Ntrain]
        Ytrain = Y[:Ntrain]

        Xtest = X_poly[Ntrain:]
        Ytest = Y[Ntrain:]

        for d in range(MAX_POLY):
            model.fit(Xtrain[:,:d+2], Ytrain)
            predictions= model.predict(X_poly[:,:d+2])

            #Visualise the predictions
            x_axis_poly = Make_Poly(x_axis, d+1)
            prediction_axis = model.predict(x_axis_poly)

            prediction_curves[:,k,d] = prediction_axis

            train_prediction = predictions[:Ntrain]
            test_prediction = predictions[Ntrain:]

            train_score = mse(train_prediction, Ytrain)
            test_score = mse(test_prediction, Ytest)

            train_scores[k,d] = train_score
            test_scores[k,d] = test_score
