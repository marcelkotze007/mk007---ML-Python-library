from scipy.stats import norm
import scipy as sc
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
"""
print(norm.pdf(0))     #Gives the PDF of 0 for a normal distribution     
print(norm.pdf(0, loc = 5, scale = 10))  #mean = loc, standard deviation = scale

R = np.random.randn(10)                  #Creates 10 normally distributed random values
print(norm.pdf(R))                       #Calculates the PDF of all the values at the same time

#Working with the log of PDF:
print(norm.logpdf(R))                    #Gives the log of the PDF 

#CDF functions:
print(norm.cdf(R))                       #Gives the CDF 
print(norm.logcdf(R))                    #Gives the log of CDF
"""
"""
#Sampling from a Gaussian distribution
R = np.random.randn(10000)
plt.hist(R, bins = 100)
plt.show()

R = 10*np.random.randn(10000) + 5        #The 10 (standard deviation) is to scale the data, the 5 is the mean
plt.hist(R, bins = 100)
plt.show()
"""
"""
#Spherical Gaussian distribution:
R = np.random.randn(10000, 2)                         #Creating data in more than 1 dimension
plt.scatter(R[:,0], R[:,1])                           #Verifies that the data is spread out in more than 1 dimension
plt.show()
#Setting the standard deviation and mean:
R[:,1] = 5*R[:,1] + 2                                 #Thus the standard deviation is 5, mean = 2
plt.scatter(R[:,0], R[:,1])
plt.axis("equal")                                     #Sets the axis to be equal
plt.show()
"""
"""
#Multivariate distribution:
#Creating a covariance matrix, with variance 1 in the first dimension and a variance of 3 in the second dimension
#The covariance between the dimension is 0.8
cov = np.array([[1, 0.8], [0.8, 3]])    

mu = np.array([0,2])                                                  #Set the mean equal to 2
#Create random data sample from a multivariate normal distribution
R = mvn.rvs(mean= mu, cov= cov, size= 1000)
#R = np.random.multivariate_normal(mean = mu, cov = cov, size = 1000) #Exactly the same as for mvn.rvs()
plt.scatter(R[:,0], R[:,1])
plt.show()
"""
"""
#Interesting functions in Scipy:
#Using Matlab, using their own file format known as .mat files
sc.io.loadmat(file_name)

#Loading in an audio file:
#Return the sample rate (in samples/sec) and data from a WAV file
sc.io.wavfile.read(filename)  #For reading
cs.io.wavfile.write(filename) #For writing

#Signal processing:
#Popular signal processing is convolution
sc.signal.convolve()

#Fourier signal processing:
#Can be found in numpy library
#Fourier processing converts a signal from the time domain to the frequency domain
#The example follows the Fourier Series in the Deep Learning, Basics notes
x = np.linspace(-20, 20, 10000)
y = 1.5 + np.sin(6/np.pi*x) + np.sin(6/(3*np.pi)*x) + np.sin(6/(5*np.pi)*x) + np.sin(6/(7*np.pi)*x)

plt.plot(y)
plt.show()
#Compute the one-dimensional discrete Fourier Transform
Y = np.fft.fft(y)
plt.plot(Y)
plt.show()
"""
