import numpy as np 
import matplotlib.pyplot as plt 
import tensorflow as tf 
from datetime import datetime as dt

N = 500
D = 2
M = 30
K = 3

#first cloud is centred at (0, -2)
X1 = np.random.randn(500, 2) + np.array([0, -2])
#second cloud is centred at (2, 2)
X2 = np.random.randn(500, 2) + np.array([2, 2])
#Third cloud is centred at (-2, 2)
X3 = np.random.randn(500, 2) + np.array([-2, 2])
X = np.vstack((X1,X2,X3))
Y = np.array([0]*N + [1]*N + [2]*N)

N1 = len(Y)

T = np.zeros((N1, K))
T[np.arange(N1), Y[:].astype(np.int32)] = 1

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev = 0.01))

def forward(X, W1, b1, W2, b2):
    #Z = tf.nn.sigmoid(tf.matmul(X, W1) + b1)
    #Z = tf.nn.tanh(tf.matmul(X, W1) + b1)
    Z = tf.nn.relu(tf.matmul(X, W1) + b1)
    return tf.matmul(Z, W2) + b2

tfX = tf.placeholder(tf.float32, [None, D])
tfY = tf.placeholder(tf.float32, [None, K])

W1 = init_weights([D,M])
b1 = init_weights([M])
W2 = init_weights([M,K])
b2 = init_weights([K])

logits = forward(tfX, W1, b1, W2, b2)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logits, labels = tfY))

train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)
predict_op = tf.argmax(logits, 1)

sessions = tf.Session()
init = tf.initialize_all_variables()
sessions.run(init)
dt0 = dt.now()
for i in range(10000):
    sessions.run(train_op, feed_dict={tfX: X, tfY: T})
    pred = sessions.run(predict_op, feed_dict={tfX: X, tfY: T})

print(np.mean(Y == pred))
print(dt.now() - dt0)

