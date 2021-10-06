import pandas as pd
import numpy as np 
from datetime import datetime as dt

from ANN_General import ANN

filename_Com = r"C:\Users\Marcel\OneDrive\Python Courses\Machine Learning\train.csv"
filename_Tab = r"C:\Users\marce\OneDrive\Python Courses\Machine Learning\train.csv"

data = pd.read_csv(filename_Tab)
#data.header()
dataframe = data.values.astype(np.float32)
np.random.shuffle(dataframe)
limit = 33600

Y = dataframe[:limit,0]
X = dataframe[:limit,1:] 
Y_test = dataframe[limit:,0]
X_test = dataframe[limit:,1:]
#Normalize the data:
mu = X.mean(axis=0)
std = X.std(axis=0)
np.place(std, std == 0, 1)
X = (X - mu) / std
X_test = (X_test - mu) / std

model = ANN(activation_function="relu",GD_method="Batch_SGD")

dt0 = dt.now()
model.fit(X, Y, epochs=10000, regularization=0.1, learning_rate=1e-7, 
M=250, goal="classification", costs_show=False, batch_size=1000, 
momentum="nesterov", optimizer="rmsprop", early_stop=0.003, batch_norm=False)
print(model.score(X_test,Y_test))
print("Time", dt.now() - dt0)

#Y_hat = model.predict(X_test)
#P = np.argmax(Y_hat, axis=1)
#print(P[400])
