import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from datetime import datetime as dt
# from perfect import task, Flow

# @task
def Get_Data(normalize = True):
    print("Reading in and transforming data...")
    # df = pd.read_csv(r"C:\Users\marcel.kotze\OneDrive - Investec\Desktop\ML_Data.csv")
    df = pd.read_csv(r"C:\Users\marcel.kotze\OneDrive - Investec\Desktop\ML_Data_Non_Zero.csv")
    # print(df.head(10))
    data = df.values
    np.random.shuffle(data)
    X1 = data[:, 3:12]
    X2 = np.array(data[:, 13:-2], dtype=np.float64)
    # print(X1)
    # print(X2)
    Y = data[:, -2]

    if normalize:
        mu = X2.mean(axis=0)
        print(mu)
        std = X2.std(axis=0)
        np.place(std, std == 0, 1)
        X = (X2 - mu) / std

    X = np.concatenate((X1,X2),axis=1)

    return X, Y

# @task
def Splitting_Data(X, Y, Split_Per = 0.95):
    Ntrain = int(len(Y) * Split_Per)
    print(f"Training dataset: {Ntrain}")
    print(f"Testing dataset: {len(Y) - Ntrain}") 

    Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
    Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

    return Xtrain, Ytrain, Xtest, Ytest, Ntrain

# @task
def Draw_Result(Yhat, Y_test):
    i = 0
    while i < len(Yhat):
        if Yhat[i] == 0: #or Yhat_L2[i] == 0 or Y_test[i] == 0:
            pass
        else:
            predict = plt.scatter(i, Yhat[i], c = '#1f77b4')
            actual = plt.scatter(i, Y_test[i], c = '#bcbd22')
        i+= 1

    predict.set_label('Y_predict_test')
    actual.set_label('Actual')
    plt.legend()
    plt.show()

# @task
def Cross_Validation(Xtrain, Ytrain, Xtest, Ytest, max_k = 30, n_seed=0):
    max_k = 30
    best_error = float(0)
    k = 0

    while k < max_k:
        model = RandomForestRegressor()
        model.fit(Xtrain, Ytrain)

        #print("train accuracy:", model.score(Xtrain, Ytrain))
        cur_error = model.score(Xtest, Ytest)
        if k % 10 == 0:  
            print("test accuracy:", cur_error)

        if cur_error > best_error:
            with open(rf'C:\Users\marcel.kotze\OneDrive - Investec\Desktop\forecast_forex_loan_{n_seed}.pkl', 'wb') as f:
                pickle.dump(model, f)
        
        k += 1

# @task
def MSE(Y, Yhat, N):
    """
    Determines the Mean Square of Error, determines how acuare the model is. The lower the better
    """
    delta = Y - Yhat
    mse = delta.dot(delta) / N
    return mse

def R_squared(Y, Yhat):
    """
    Calculates the R squared that indicates how well the model fits the data.
    The closer R squared is to 1, the better the fit. 
    Must first calculate the expected values of Y i.e. Yhat, using calculate_a_b() function
    """
    dif1 = Y - Yhat
    dif2 = Y - Y.mean()
    
    #Rater use the dot function as it multiplies and sums the values, vector multiplication,
    #as 100x1 * 1x100 = a single value, the sum                                    
    R = 1 - dif1.dot(dif2)/dif2.dot(dif2) 
    return R

def Test_Formula_Accuracy():
    df = pd.read_csv(r"C:\Users\marcel.kotze\OneDrive - Investec\Desktop\Formula_Accuracy.csv")
    # print(df.head(10))
    data = df.values
    Y = data[:, -2]
    Yhat = data[:, -1]

    R = R_squared(Y, Yhat)

    print(f"The accuracy of the formula is {R}")

# @task
def Training():
    seed = 0
    while seed < 20:
        X, Y = Get_Data()
        Xtrain, Ytrain, Xtest, Ytest, Ntrain = Splitting_Data(X, Y)
        Cross_Validation(Xtrain, Ytrain, Xtest, Ytest, n_seed=seed)
        seed += 1

    i = 0
    best_score = 0.5
    best_seed = 0
    best_mse = 0
    while i < seed-1:
        try:
            with open (rf'C:\Users\marcel.kotze\OneDrive - Investec\Desktop\forecast_forex_loan_{i}.pkl', "rb") as f:
                model = pickle.load(f) #already trained

            Y_predict_test = model.predict(Xtest)

            mse = MSE(Ytest, Y_predict_test, (len(Y) - Ntrain))
            error = model.score(Xtest, Ytest)
            print("R-squared: ", error)
            print("Mean Square Error: ", mse)

            if error > best_score:
                best_score = error
                best_mse = mse
                best_seed = seed
            else: 
                os.remove(rf'C:\Users\marcel.kotze\OneDrive - Investec\Desktop\forecast_forex_loan_{i}.pkl')
        except:
            pass
        i+=1

    return best_seed, Xtest, best_score, best_mse, Ytest

# @task
def Training_Val(best_seed, Xtest, best_score, best_mse, Ytest):
    with open (rf'C:\Users\marcel.kotze\OneDrive - Investec\Desktop\forecast_forex_loan_{best_seed}.pkl', "rb") as f:
        model = pickle.load(f) #already trained
    
    Y_predict_test = model.predict(Xtest)
    print("R-squared: ", best_score)
    print("Mean Square Error: ", best_mse)
    
    Draw_Result(Y_predict_test, Ytest)

# @task
def Random_Test(best_seed, Split_Per=0, stats = False, draw = False):
    X, Y = Get_Data()
    _, _, Xtest, Ytest, Ntrain = Splitting_Data(X, Y, Split_Per=Split_Per)

    with open (rf'C:\Users\marcel.kotze\OneDrive - Investec\Desktop\forecast_forex_loan_{best_seed}.pkl', "rb") as f:
        model = pickle.load(f) #already trained
    then = dt.now()
    Y_predict_test = model.predict(Xtest)

    if stats:
        print("Time to make predictions: ", str(dt.now()-then))
        mse = MSE(Ytest, Y_predict_test, (len(Y) - Ntrain))
        error = model.score(Xtest, Ytest)
        print("R-squared: ", error)
        print("Mean Square Error: ", mse)
    
    if draw:
        Draw_Result(Y_predict_test, Ytest)

if __name__ == '__main__':

    # Build a flow of the process
    # with Flow("Testing_Flow") as flow:

    #Building the Model and testing the model
    best_seed, Xtest, best_score, best_mse, Ytest = Training()
    Training_Val(best_seed, Xtest, best_score, best_mse, Ytest)
    # Random_Test(4, Split_Per=0.0, stats = True, draw = True)
    # Test_Formula_Accuracy()

    # flow.run()
