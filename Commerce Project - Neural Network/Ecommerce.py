from sklearn.utils import shuffle
from datetime import datetime as dt

from Pre_proccessing_Data import proccessing_data
from ANN_General import ANN

filename = "ecommerce_data.csv"
pd = proccessing_data()

# get the data
X, Y = pd.get_data(filename)

X, Y = shuffle(X, Y)
Ntrain = int(0.7 * len(X))
Xtrain, Ytrain = X[:Ntrain], Y[:Ntrain]
Xtest, Ytest = X[Ntrain:], Y[Ntrain:]

# create the neural network
model = ANN(activation_function='tanh')

dt0 = dt.now()
# train the neural network
model.fit(Xtrain, Ytrain,epochs=2000, regularization=0.001, learning_rate=0.0001, M=20, goal="classification", costs = False)

# print the train and test accuracy
train_accuracy = model.score(Xtrain, Ytrain)
test_accuracy = model.score(Xtest, Ytest)
print("train accuracy:", train_accuracy, "test accuracy:", test_accuracy)

print(dt.now() - dt0)