import numpy as np
import matplotlib.pyplot as plt

#Generate the data using discrete uniform distribution:
N = 10
rawdata = np.random.random((N, 2)) 
data = rawdata * 2 - 1      #All values > 0.5, will be positive, while values < 0.5 will be made negative
#print(data)
#Generate the labels, the 4 sectors that will be created 
Y = np.zeros(N)   #create a empty nparray
Y[(data[:,0] < 0) & (data[:, 1] > 0)] = 1  #Creates the first two sectors, row 1
#Takes all the pos values from col 1 and all of the neg values from col 0
Y[(data[:,0] > 0) & (data[:, 1] < 0)] = 1  #Creates the second two sectors, row 2
#Takes all of the neg values of col 1 and all of the pos values of col 0

#Plotting the graph:
plt.scatter(data[:, 0], data[:, 1], c = Y)
plt.show()
