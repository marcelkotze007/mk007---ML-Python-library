import numpy as np 

import tensorflow as tf 

#As with theano, one must first specify the type
A = tf.placeholder(tf.float32, shape=(5,5), name='A')

#Shape and name are optional
v = tf.placeholder(tf.float32)

#Instead of dot, we use matmul --> matrix multiplication
w = tf.matmul(A, v)

#Similar to Theano, need to 'feed' the variables values
#In TensorFlow you do the actual work in a session
# 
with tf.Session() as session:
    # The values are fed in via the appropriately named argument 'feed_dict' 
    # V needs to be of shape=(5, 1) not just shape=(5,)
    # It's more like 'real' matrix multiplication
    output = session.run(w, feed_dict={A: np.random.randn(5,5), v: np.random.randn(5,1)})

    # What's this output that is returned by the session
    print(output, type(output))

    #The output array should just be a numpy array

# TensorFlow variables are like Theano shared variables
# But Theano variables are like TensorFlow placeholders 

# A TensorFlow variable can be initialized with a numpy array or a TF array
# Or more correctly, anything that can be turned into a TF tensor
shape = (2,2)
x = tf.Variable(tf.random_normal(shape))
# x = tf.Variable(tf.random_normal(2,2))
t = tf.Variable(0) #Creates a scalar

# Need to initialize the variables first
init = tf.global_variables_initializer()

with tf.Session() as session:
    out = session.run(init) #Runs the init operation
    print(out) #Should just print out None

    # eval() in TF is like get_value() in Theano
    print(x.eval()) # Gets the initial value of x
    print(t.eval())

# Example of finding the min of a simple cost function, like in Theano example (Theano1.py)
u = tf.Variable(20.0)
cost = u*u + u + 1.0

# One difference between Theano and TF is that you don't write the updates yourself in TF
# You choose an optimizer that implements the algorithm you want
# 0.3 is the learning_rate. The documentation lists the parameters
train_op = tf.train.GradientDescentOptimizer(0.3).minimize(cost) #Just normal gradient descent

# Run a session again:
init = tf.global_variables_initializer()
with tf.Session() as session:
    session.run(init)

    # Strangely, while the weight update is automated, the loop itself is not
    # So we'll just call train_op until convergence
    # This is useful for us anyway since we want to track the cost function
    for i in range(12):
        session.run(train_op)
        print("i = %d, cost = %.3f, u = %.3f" %(i, cost.eval(), u.eval()))
 

