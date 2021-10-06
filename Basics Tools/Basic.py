import numpy as np
from datetime import datetime as dt

L = [1,2,3]
#A numpy array is basically a matrix
A = np.array([1,2,3])
print("%s \n%s" %(A, L))
"""
for e in L:
    print(e)
for e in A:
    print (e)
"""
"""
#These are 2 methods for appending a list
#Adds a 4 and a 5 to the previously created list
L.append(4)
L = L + [5]
print(L)

L2 = []
for e in L:
    L2.append(e + e)
print (L2)
L2 = []
for e in L:
    L2.append(e**2)
print (L2)
"""
"""
#When using numpy it is best used to represent vectors: 
print(A + A) #The + does vector addition
print(2 * A) #The * does vector multiplication
print(A**2) #Can be directly used 
print(np.sqrt(A))
print(np.log(A))
print(np.exp(A))
"""
"""
a = np.array([1,2])
b = np.array([2,1])
dot = 0
for e, f in zip (a, b):
    dot += e*f #The += is used to accumalate the data
print (dot)
#Multiplication of 2 arrays
print(a*b) #produces an element wise production, which means the 2 matrixes must be the same size
#Using a sum function one can multiply vectors to obtain a skalaar
print(np.sum(a*b))
print((a*b).sum())
#Using the dot function
print(np.dot(a,b))
print((a).dot(b))
print((b).dot(a))
#Calculating the length of a the vector
amag = np.sqrt((a**2).sum())
print(amag)
amag = np.linalg.norm(a)
print(amag)
#Calculating the angle of a vector in radians:
cosangle = a.dot(b)/(np.linalg.norm(a) * np.linalg.norm(b))
print(np.arccos(cosangle))
"""
#Speedtest between for loop and numpy 
a = np.random.randn(100)
b = np.random.randn(100)
T = 100000

def slow_dot_product(a, b):
    res = 0
    for e, f in zip(a, b):
        res += e*f
    return res

t0 = dt.now()
for t in range(T):
    slow_dot_product(a, b)
dt1 = dt.now() - t0

t0 = dt.now()
for t in range(T):
    a.dot(b)
dt2 = dt.now() - t0

print("dt1 / dt2 :", dt1.total_seconds() / dt2.total_seconds())

"""
Important functions to know for gradient decent:
N = 10
D = 3  #Dimensionality
X = np.zeros((N,D))

X[:, 0] = 1   #bais column
X[:5, 1] = 1  #Sets the first five rows of column 2 to 1
X[5:, 2] = 1  #Sets the last five rows of column 3 to 1

#print(X)

Y = np.array([0] * 5 + [1] * 5)  #Sets Y = 0, for first half of the data and Y = 1 for the second half of the data

print(Y)
"""