import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import the data and store as a nparray:
dataset = pd.read_csv("C:/Users/Marcel/OneDrive/Python Courses/Deep Learning/train.csv").values

#Shuffle the images:
np.random.shuffle(dataset)

#Must first shuffle the data then plot the X and Y values
X = dataset[:, 1:]
Y = dataset[:, 0]   #Labels for the data

#Looking at to rotate functions:
#1. Is a built in function of numpy:
def rotate1(image):
    return np.rot90(image, 3)

#2. Is through a for loop that rotates each value one by one:
def rotate2(image):
    H, W = image.shape
    image2 = np.zeros((W, H))
    for i in range(H):
        for j in range(W):
            image2[j, H - i - 1] = image[i,j]
    return image2

#Calling the functions to flip the image
for i in range(X.shape[0]):
    #First get the image by reshaping the shape of the ndarray
    image = X[i].reshape(28, 28)

    #Flip the image using the functions created:
    #image = rotate1(image)  #The function using the built in np.rot90() function
    image = rotate2(image)   #The function using the for loop

    #Lastly need to plot the image:
    plt.imshow(255 - image, cmap= 'gray')
    plt.title("Label: %s" %Y[i])
    plt.show()

    answer = input("Continue? [Y/n]: ")
    if answer and answer[0].lower() == 'n':
        break

