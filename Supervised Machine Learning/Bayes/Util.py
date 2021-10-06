import numpy as np 
import pandas as pd    
    
def get_data(limit = None, filename = "C:/Users/Marcel/OneDrive/Python Courses/Machine Learning/train.csv"):
    """
    Reads the MNIST dataset and outputs X and Y.
    One can set a limit to the number of rows (number of samples) by editing the 'limit'
    """
    print("Reading in and transforming data...")
    dataset = pd.read_csv(filename).values
    np.random.shuffle(dataset)
    X = dataset[:, 1:] / 255
    Y = dataset[:, 0]
    if limit is not None:
        X, Y = X[:limit], Y[:limit]
    print("Done reading in data...", len(Y))
    return X, Y
