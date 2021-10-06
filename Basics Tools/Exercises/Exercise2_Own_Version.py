import numpy as np
import matplotlib.pyplot as plt

#Custom function that generates a random uniform distribution
def sampleZ(n = 10000):
    X = np.random.random(n)
    Y = X.sum()              #sums the 10000 variables
    return Y

N = 10000                    #loops trough the function 10000 times
Y_samples = np.zeros(N)      #Create an empty array
for i in range(N):           #Loops trough the empty array and ads the random distribution into the array
    Y_samples[i] = sampleZ() #Thus, creating 10000 random sums of uniform distributed data

mean = Y_samples.sum()/N
values = np.zeros(N)
for i in range(N):
    values[i] = (sampleZ() - mean)**2

variance = values.sum()/N
print("Mean = %s \nVariance = %s" %(mean, variance))

#Plot the values to illistrate that the central limit theory holds true
plt.hist(Y_samples, bins = 100)
plt.show()