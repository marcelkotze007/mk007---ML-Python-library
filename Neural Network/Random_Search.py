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

#Random search

model = ANN(activation_function="relu",GD_method="Batch_SGD")

dt0 = dt.now()
learn = -6
reg = -2
hidden_size = 200
max_tries = 30

best_cost = 0
best_learn = None
best_reg = None
best_hidden = None

for _ in range(max_tries):
    model.fit(X, Y, epochs=10000, regularization=10**reg, learning_rate=10**learn, 
    M=hidden_size, goal="classification", costs_show=False, batch_size=1000, 
    momentum="nesterov", optimizer="rmsprop", early_stop=0.002, batch_norm=False)
    cost = model.score(X_test,Y_test)
    print("learn:%f reg:%f hidden:%s cost:%f" %(learn, reg, hidden_size, cost))
    if cost > best_cost:
        best_cost = cost
        best_hidden = hidden_size
        best_reg = reg
        best_learn = learn
    
    #Select new hyperparams:
    hidden_size = best_hidden + np.random.randint(-1,4) * 10
    hidden_size = max(10, hidden_size)
    learn = best_learn + np.random.randint(-1,2)
    learn = min(-5, learn)
    reg = best_reg + np.random.randint(-1,2)


print("Best Cost:", best_cost)
print("Best Hidden:", best_hidden)
print("Best regular:", best_reg)
print("Best learning rate:", best_learn)

print("Time", dt.now() - dt0)