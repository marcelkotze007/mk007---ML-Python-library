import numpy as np

M = np.array([[1,2], [3,4]])
print(M)
"""
print(M[0][0])
print("Using shorthand to extract the first value of a matrix:", M[0,0])
#When using lists for ex. L = [[1,2],[3,4]]
#Would use L[0][0] to get the value 1
L = [[1,2],[3,4]]
print("Print the forth value in the matrix:", L[1][1])
"""
"""
M2 = np.matrix([[1,2],[3,4]])
#CONVERT MATRIX TO ARRAYS!!!!!!!!!!!!!!!!!!!!!!!
M2 = np.array(M2)
#print (type(M2))
print(M2)
"""
"""
#Creating arrays containing only zeros:
Z = np.zeros(10)
MZ = np.zeros((10,10))
print(Z)
print(MZ)

#Creating arrays containing only 1:
one = np.ones((10, 10))
print(one)

#Creating arrays containing uniformly distributed random numbers between 0 and 1:
rand = np.random.random((10,10))
print(100*rand)

#Creating arrays containing random numbers from a normal distribution:
Gaus = np.random.randn(10,10) #only for randn one must pass each dimension seperatly
print(Gaus)
print(Gaus.mean())
print(Gaus.var())
"""
"""
#Matrix production:
X = np.random.rand(3,2)
Z = np.random.rand(2,3)
C = Z.dot(X)
print ("%s \n%s \n%s" %(X,Z,C))

#Creating a matrix inverse:
A = np.array([[1,2],[3,4]])
Ainv = np.linalg.inv(A)
print(Ainv)

#Finding the identity matrix:
IdentA = Ainv.dot(A)
print (IdentA)
IdentA = A.dot(Ainv)
print(IdentA)

#Calculating the determinant:
np.linalg.det(A)

#Gives the diagonal elements:
np.diag(A)

#Can also construct a matrix by entering a 1D array and the using those values as the diagonal:
np.diag([1,2])
#Thus diag works inversly if given a 1D array it will create a 2D array, and if given a 2D array will construct a 1D array

a = np.array([1,2])
b = np.array([3,4])

np.outer(a, b)
np.inner(a, b) #Is the same as a dot product

#Matrix trace is the sum of the diagonals of a matrix:
np.diag(A).sum()
np.trace(A)
"""
"""
#Eigenvalues and Eigenvectors - Used when a matrix is not static 
X = np.random.randn(100, 3)
cov = np.cov(X) 
print("If the shape is 100, 100 it is incorrect: ", cov.shape) #Check to make sure the correct shape is given 
cov = np.cov(X.T) #When calculating the covariance of a matrix it needs to be transposed first
print(cov)

#np.linalg.eig(X) # Use this for most calculations
eigh = np.linalg.eigh(cov) # Use this for symmetrical matrix or Hermitian matrices (A = A.T = A.H)
print(eigh) #The eigh functions gives 2 tuples, the first is the eigenvalues and the second is the eigenvectors
"""
"""
#Solving a linear system
B = np.array([1,2])
x = np.linalg.inv(M).dot(B)
print(x)
x = np.linalg.solve(M, B) #Always use the solve method to solve linear equations as it is more efficient
print(x)
"""

#Word Problem
E1 = np.array([[1, 1], [1.5, 4]])
E2 = np.array([2200, 5050])
sol = np.linalg.solve(E1, E2)
print(sol)
