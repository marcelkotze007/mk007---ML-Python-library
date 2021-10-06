from Pre_proccessing_Data import proccessing_data
from datetime import datetime as dt

from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle

filename = "ecommerce_data.csv"
pd = proccessing_data()

# get the data
X, Y = pd.get_data(filename)

X, Y = shuffle(X, Y)
Ntrain = int(0.7 * len(X))
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

# create the neural network
model = MLPClassifier(hidden_layer_sizes=(20), max_iter=2000)

dt0 = dt.now()
# train the neural network
model.fit(Xtrain, Ytrain)

# print the train and test accuracy
train_accuracy = model.score(Xtrain, Ytrain)
test_accuracy = model.score(Xtest, Ytest)
print("train accuracy:", train_accuracy, "test accuracy:", test_accuracy)

print(dt.now() - dt0)