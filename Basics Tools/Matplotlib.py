import matplotlib.pyplot as plt
import seaborn
import numpy as np
import pandas as pd

"""
#A simple graph 
x = np.linspace(0, 10, 1000)        #Used to generate data, has 3 data points: Start, End, Number of points
y = np.sin(x)                       #Creates a sin wave

plt.plot(x,y)                       #Plots the sin wave

plt.xlabel("Time")                  #Adds a x label
plt.ylabel("Some function of time") #Adds a y label
plt.title("My new Sin Chart")       #Adds a title to the graph

plt.show()                          #To see the graph, it must be shown
"""
"""
#Scatterplot:
#First the data must be loaded in and converted to a numpy array
A = pd.read_csv("data_1d.csv", header= None).values #When using .values instead of .as_matrix, the values has no ()
print(type(A))

#Creating the data:
#The colon indicates that all values must be selected in a row, but only a single column must be selected
x = A[:,0]                                           
y = A[:,1]                                           #Setting up the x and y axis of the scatter plot

plt.scatter(x,y)                                     #Creates the scatter graph

#Inserting fitting line:
x_line = np.linspace(0, 100, 100)                    #Creates x-points
y_line = 2*x_line + 1                                #Creates the y-points

plt.plot(x_line, y_line)
plt.show()
"""
"""
#Plotting a histogram from data:
A = pd.read_csv("data_1d.csv", header= None).values  #Loading in the data into python
x = A[:,0]
y = A[:,1]

plt.hist(x, bins = 20)                               #Creates the histogram, bins = number of histograms
plt.show()

#Shows that the data is normally distributed
y_actual = 2*x + 1
residuals = y - y_actual
plt.hist(residuals)
plt.show()
"""
"""
#Plotting uniformly distributed random variables
R = np.random.random(10000)
plt.hist(R)
plt.show()
#Plotting a normal/Gaussian distribution 
N = np.random.randn(10000)
plt.hist(N)
plt.show()
"""
"""
#Read in the data from the train.csv file
train = pd.read_csv("C:/Users/Marcel/OneDrive/Python Courses/Deep Learning/train.csv")
print(train.head())      #Checks that data imported correctly
data = train.values      #Converts data into a numpy array

#Grab an image from the dataset:
image = data[0, 1:]      #Select the 0 row, give all columns except for column 0, as that is not a pixel
print(image.shape)       #Shape function does not have (), shows the vector

#Reshape the data into a 28x28 image:
image = image.reshape(28, 28)  #Reshapes the image from a 784 to a 28x28
print(image.shape)             #Verifies that the image has been reshaped

#Plot image:
#The 225 - is to inverse the colour, so the image is black and the background is white
plt.imshow(255 - image, cmap= 'gray')   #Creates the image, the cmap = determines the color of the image
plt.show()                              #Shows the image

#Check the label in the dataset to determine what the image is supposed to be:
print("The image is a:", data[0,0])
"""