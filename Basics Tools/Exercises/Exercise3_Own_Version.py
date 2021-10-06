import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Loading the data into python using pandas, and converts it into a nparray
dataset = pd.read_csv("C:/Users/Marcel/OneDrive/Python Courses/Deep Learning/train.csv").values
X = dataset[:, 1:] #Imports the images
Y = dataset[:, 0]  #Imports the labels

#Create a for loop to cycle trough all the labels:
for label in range(10):
    Xlabel = X[Y == label]              #Creates a label for each X value, by cycling trough the data labels 0-9
    
    #Calculate the mean of the images:
    Meanlabel = Xlabel.mean(axis = 0)   #Need to specify the axis as there are more than 1-D
    
    #Next reshape as an image:
    image = Meanlabel.reshape(28, 28)

    #Plotting the image:
    plt.imshow(255 - image, cmap= 'gray')
    plt.title("Label: %s" %label)
    plt.show()

#Program shows the mean image for each of the labels given, from 0-9