import numpy as np
import pandas as pd
from sklearn.utils import shuffle

filename = ("D:/Machine Learning datasets/"
"challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013.csv")
filename_test = ("D:/Machine Learning datasets/"
"challenges-in-representation-learning-facial-expression-recognition-challenge/fer2013/fer2013_test.csv")

def init_weights_and_bias(self, N, D):
    """
    Intitializes the intial random Gaussian ditributed weights and bias. Using the length N and the number of
    features D. Also normalizes the weights. Converts the output to format float32 in order to use them with
    theano and tensorflow. 
    """
    W = np.random.randn(N,D) / np.sqrt(N)
    b = np.zeros(D)

    return W.astype(np.float32), b.astype(np.float32)

def init_filter(self, shape, poolsz):
    """
    Used for convolutional neural networks. Shape is a typle of 4 different values. Divide by something like the 
    fan in, fan out, in order to obtain the normalized value. Converts the output to format float32 in order to 
    use them with theano and tensorflow.
    """
    w = np.random.randn(*shape) * np.sqrt(2) / np.sqrt(np.prod(shape[1:]) + shape[0] * np.prod(shape[2:] / np.prod(poolsz)))
    return w.astype(np.float32)

def relu(self, x):
    """
    This is the rectifier linear unit that is used for activition function inside the neural network.
    Use for older versions of theano that doesn't have it buildt in
    """
    return x * (x > 0)

def sigmoid(self, z):
    """
    Takes in argument z which is the dot porduct of X_bias.dot(w) and returns the expected value (Y_hat/Y) of
    the targets (Y/T). Thus, making a prediction based on the weights and bias
    """
    return 1 / (1 + np.exp(-z))

def softmax(self, z):
    """
    Forms part of deep learning
    """
    expA = np.exp(z)
    return expA/expA.sum(axis = 1, keepdims = True)

def cross_entropy_error(self, Y, Y_hat):
    """
    Calculates the cost function of binary models and ,thus indicates how 'wrong' our model is, the closer
    the cost is to 0 the more accurate the model is. The larger the cost, the lower our accuracy is. 
    """
    return -np.mean(Y * np.log(Y_hat) + (1 - Y) * np.log(1 - Y_hat))

def cost(self, Y, Y_hat):
    """
    A more general cross entropy cost function and can be untilized in softmax. Also known as the 
    the softmax cross entropy. Thus, returns the cost and accuracy of the model
    """
    return -(Y * np.log(Y_hat)).sum()

def cost_alt(self, Y, Y_hat):
    """
    The same as cost(), just uses the targets to index Y.
    Instead of multiplying by a large indicator matrix with mostly 0s
    """
    N = len(Y)
    return -np.log(Y[np.arange(N), Y_hat]).mean()

def classification_rate(self, targets, predictions):
    """
    Returns the accuracy of the model by indicating how many times the model correctly classified the
    predicted outcome
    """
    return np.mean(targets == predictions)

def error_rate(self, targets, predictions):
    """
    The inverse of the classification rate, thus returns the average amount of incorrect predictions the model
    made and showcases the error %
    """
    return np.mean(targets != predictions)

def y2_indicator(self, y):
    """
    Turns a Nx1 vector of targets (class labels 0 to k-1) into an indicator matrix that will consit of 
    only zero and ones, but the size will be NxK
    """
    N = len(y)
    K = len(set(y))
    index = np.zeros((N, K))
    for i in range(N):
        index[i, y[i]] = 1
    return index

def get_data(self, balance_ones = True):
    """
    Extracts the data from a csv file and saves it as a nd.array. Also separates the data into X-values 
    and Y-targets. Also adjusts data (normalize the data) to be between 0 and 1. 
    Images are 48x48 = 2304 size vectors. Because classes are imbalanced, class 1 is lengthend 9 times.
    """
    #dataframe = pd.read_csv(filename)
    #dataset = dataframe.values
    #Y = dataset[:, 0]
    #X = dataset[:, 1:-1]
    #print(X[:10])
    #X = dataset[:, 1:-1] * 1/255.0
        # images are 48x48 = 2304 size vectors
    Y = []
    X = []
    first = True
    for line in open(filename_test):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X), np.array(Y)
    
    mu = X.mean(axis=0)
    std = X.std(axis=0)
    np.place(std, std == 0, 1)
    X = (X - mu) / std


    if balance_ones:
        #All the data that is not class one is extracted and placed in X0 and Y0
        X0, Y0 = X[Y != 1,:], Y[Y != 1]
        #X1 contains all of the data from class 1 and is then repeated 9 times 
        X1 = X[Y == 1, :]
        X1 = np.repeat(X1, 9, axis = 0)
        X = np.vstack([X0, X1])
        Y = np.concatenate((Y0, [1] * len(X1)))

    return X, Y

def get_image_data(self):
    """
    Call the get_data() and then reshapes the data and returns new arrays X and Y. Will use this function
    when using convolutional neural networks. Keeps the orginal image shape
    """
    X, Y = get_data(filename_test)
    N, D = X.shape
    d = int(np.sqrt(D)) #determines the length == width
    X = X.reshape(N, 1, d, d)

    return X, Y

def get_binary_data(self):
    """
    Extracts the data from the csv file and returns X and Y between 0 and 1.Y values are 0 or 1
    """
    Y = []
    X = []
    first = True
    for line in open(filename_test):
        if first:
            first = False
        else:
            row = line.split(',')
            y = int(row[0])
            if y == 0 or y == 1:
                Y.append(y)
                X.append([int(p) for p in row[1].split()])
    return np.array(X) / 255.0, np.array(Y)

def cross_validation(self, model, X, Y, K = 5):
    """
    Nan
    """
    X, Y = shuffle(X, Y)
    sz = len(Y) // K #sample size
    errors = []
    for k in range(K):
        X_train = np.concatenate([X[:k * sz, :], X[(k * sz + sz):] ])
        Y_train = np.concatenate([X[:k * sz], Y[(k * sz + sz):] ])
        X_test = X[k * sz: (k * sz + sz), :]
        Y_test = X[k * sz: (k * sz + sz)]

        model.fit(X_train, Y_train)
        err = model.score(X_test, Y_test)
        errors.append(err)
    print('Errors: ', errors)

    return np.mean(errors)

    