import numpy as np
import matplotlib.pyplot as plt

#Built in numpy function that test if a matrix is symmetrical by comparing it to its transpose
def is_symmetrical1(A):
    return np.all(A == A.T)

#Manual way to assirtain if a matrix is symmetrical
def is_symmetrical2(A):
    rows, cols = A.shape
    if rows != cols:
        return False
    
    for i in range(rows):
        for j in range(cols):
            if A[i,j] != A[j,i]:
                return False
    return True

#Calling the functions to test a matrix:
def check(A, b):
    print("\nTesting:", A)
    assert(is_symmetrical1(A) == b)   #Statement is a debugging aid that tests a condition
    assert(is_symmetrical2(A) == b)   #Condition evaluates to false, it raises an AssertionError exception

#Test matrices:
A = np.zeros((3, 3))
check(A, True)

A = np.eye(3)   #Creates an identity matrix
check(A, True)

A = np.random.randn(3, 2)
A = A.dot(A.T)
check(A, True)

A = np.array([
    [1, 2, 3],
    [2, 4, 5],
    [3, 5, 6]
])
check(A, True)

#All of these matrices are not symmetrical and thus should produce a False
A = np.random.randn(3, 2) #When comparing matrices of different sizes the elemtwise comparison will fail
check(A, False)

A = np.random.randn(3, 3)
check(A, False)

A = np.arange(9).reshape(3, 3)
check(A, False)