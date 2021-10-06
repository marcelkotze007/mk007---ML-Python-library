import numpy as np
import matplotlib.pyplot as plt

A = np.array([       #Creates a 9x9 array
    [0.3, 0.6, 0.1], 
    [0.5, 0.2, 0.3],
    [0.4, 0.1, 0.5]
])
V = np.ones(3) / 3    #Creates a array of ones/three

print(A)              #Tests if the matrix imported correctly
print(V)              #Tests if the correct array was created

number_iterations = 25                     #Number of iterations
distances = np.zeros(number_iterations)    #Creates a new array with all 0 values

for i in range(number_iterations):          #Creates a for loop to iterate 25 times, thus creating 25 values
    answer = V.dot(A)                       #The answer is the dot product of the matrices V and A is a 1x3 matrix (also known as v-prime)
    print(answer)                            
    Euclidean = np.linalg.norm(answer - V)  #Calculates the Euclidean distance between 2 points
    distances[i] = Euclidean                #Saves the Euclidean distance to the distance array
    V = answer                              #indicates that as the iterations get larger(V = answer)

print(distances)
plt.plot(distances)
plt.show()

#Thus by saying  that answer (v-prime) = V = VA -> found the eigenvector for A for which the corresponding eigenvalue is 1