import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from Exercise8_Own_Version import create_spiral

#Get the data from the exercise:
X, Y = create_spiral()

#Combine the data into one array:
#   Data that is to be concatenated must have same number of dimensions:
#       e.g. N x D and N x 1
#       not  N x D and N
data_con = np.concatenate((X, np.expand_dims(Y, 1)), axis = 1)

dataframe = pd.DataFrame(data_con)
dataframe.columns = ['x1', 'x2', 'Y'] #Naming each column
dataframe.to_csv("Test_Data.csv", index=False)
