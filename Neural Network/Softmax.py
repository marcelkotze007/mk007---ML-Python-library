import numpy as np 

def softmax_single_array(N = 5):
    """
    Returns the probability of each of the output nodes
    """
    #Final output of a neural network
    a = np.random.randn(5)
    # These values represent the output of the output neurons
    #start by exponetiating the values:
    exp_a = np.exp(a)
    prob_answer_sin = exp_a / exp_a.sum()

    return prob_answer_sin

def softmax_matrix(N = 100, D = 5):
    """
    Returns the probability of each of the output nodes for a matrix of output 
    """
    #100 samples in 5 classes
    a = np.random.randn(N, D)
    
    exp_a = np.exp(a)
    
    #Use this to sum along the rows, the keepdims is so (100,5) (100,) can be added
    prob_answer_mat = exp_a / exp_a.sum(axis = 1, keepdims = True)

    #shows that each row sums to 1
    print(prob_answer_mat.sum(axis = 1))
