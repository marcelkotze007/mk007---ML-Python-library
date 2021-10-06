import pandas as pd
import numpy as np 
from datetime import datetime as dt

from ANN_General import ANN

data = pd.read_csv(r"C:\Users\Marcel\OneDrive\Python Courses\Machine Learning\train.csv")
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

#Grid search

model = ANN(activation_function="relu",GD_method="Batch_SGD")

dt0 = dt.now()
learn = [1e-4, 1e-5, 1e-6, 1e-7]
reg = [0.1, 0.01, 0.001]
hidden_size = [200, 300, 400]
best_cost = 0
best_learn = None
best_reg = None
best_hidden = None

for l in learn:
    for r in reg:
        for h in hidden_size:
            model.fit(X, Y, epochs=10000, regularization=r, learning_rate=l, 
            M=h, goal="classification", costs_show=False, batch_size=1000, 
            momentum="nesterov", optimizer="rmsprop", early_stop=0.002, batch_norm=False)
            cost = model.score(X_test,Y_test)
            print("learn:%f reg:%f hidden:%i cost:%f" %(l, r, h, cost))
            if cost > best_cost:
                best_cost = cost
                best_hidden = h
                best_reg = r
                best_learn = l

print("Best Cost:", best_cost)
print("Best Hidden:", best_hidden)
print("Best regular:", best_reg)
print("Best learning rate:", best_learn)

print("Time", dt.now() - dt0)