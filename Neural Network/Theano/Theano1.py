import theano.tensor as T 

#Some different types of variables
c = T.scalar('c') #Pass in the name as the first parameter
v = T.vector('v')
A = T.matrix('A')

#Can define matix multiplication
w = A.dot(v)

#Defining the values for these variables
import theano

matrix_times_vector = theano.function([A,v], w)
#Can also be initialized as follows
#matrix_times_vector = theano.function(inputs=[A,v], outputs=w)

#Importing Numpy to create real arrays
import numpy as np 
A_val = np.array([[1,2], [3,4]])
v_val = np.array([5,6])

w_val = matrix_times_vector(A_val,v_val)
print(w_val)

#In theano regular variables are not updateable need to create a shared variable
#Create a shared variable so we can do gradient descent
#This adds another layer of complexity to the theano function

x = theano.shared(20.0, 'x')
#The first argument is its initial value, second is its name

#A cost function that has a minimum value
cost = x*x + x + 1

learning_rate = 0.3
#In theano, there is no need to calculate gradients yourself
x_update = x - learning_rate * T.grad(cost, x)

#X is not an input, it's a thing you update
#In later examples, data and labels would go into the inputs
#and model parameters would go in the updates
#Updates takes in a list of tuples, each tuple has 2 things in it:
    #1) The shared variables to update
    #2) The update expression 
train = theano.function(inputs=[], outputs=cost, updates=[(x, x_update)])
#The update function, the first variable is what is to be updated and the second variable is the update function to use
#It has no input arguments

#Write a loop to call the training function
#There are no arguments
for _ in range(25):
    cost_val = train()
    print(cost_val)

#print the optimal value of x
print(x.get_value())
