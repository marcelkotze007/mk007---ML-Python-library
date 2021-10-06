import numpy as np 
import random 
import matplotlib.pyplot as plt

import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import dendrogram, linkage

#Convert list of integers to corresponding letters
def to_code(a):
    return [code[i] for i in a]

#Distance between 2 DNA strands
def dist(a, b):
    return sum(i != j for i, j in zip(a, b))

#Generate offspring by modifying some characters in the code
def generate_offspring(parent):
    return [maybe_modify(c) for c in parent]

#Modify letter c with probability ~1/1000
def maybe_modify(c):
    if np.random.random() < 0.001:
        return np.random.choice(code)
    return c

def create_ancestors():
    p1 = to_code(np.random.randint(4, size = 1000))
    p2 = to_code(np.random.randint(4, size = 1000))
    p3 = to_code(np.random.randint(4, size = 1000))

    return p1, p2, p3

def create_offspring():
    num_generations = 99
    max_offspring_per_generation = 1000
    p1, p2, p3 = create_ancestors()
    current_generation = [p1,p2,p3]

    for i in range(num_generations):
        next_generation = []
        for parent in current_generation:
            #each parent will have between 1 and 3 children
            num_offspring = np.random.randint(3) + 1

            #generate the offspring:
            for _ in range(num_offspring):
                child = generate_offspring(parent)
                next_generation.append(child)
        
        current_generation = next_generation

        #limit the number of offspring
        random.shuffle(current_generation)
        current_generation = current_generation[:max_offspring_per_generation] 

        print("Finished creating generation %d / %d, size = %d" %(i+1, num_generations, len(current_generation)))
    
    return current_generation

#Create distance matrix
#note: can use scipy's pdist for this
def create_dist_matrix():
    current_generation = create_offspring()
    N = len(current_generation)
    dist_matrix = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            elif j > i:
                a = current_generation[i]
                b = current_generation[j]
                dist_matrix[i,j] = dist(a, b)
            else:
                dist_matrix[i,j] = dist_matrix[j,i]

    dist_array = ssd.squareform(dist_matrix)

    return dist_array

def H(X, name):
    print(name)
    Z = linkage(X, name)
    plt.title(name)
    dendrogram(Z)
    plt.show()

if __name__ == "__main__":
    #Our genetic code
    code = ["A", "T", "C", "G"]
    methods = {1:'ward', 2:'single', 3:'complete'}

    dist_array = create_dist_matrix()

    for value in methods.values():
        method = value
        H(dist_array, method)


    