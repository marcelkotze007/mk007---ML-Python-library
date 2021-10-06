import numpy as np

w = random vector #randomly intialise the weights, w = np.random.randn(N, D) 
b = 0  
for epoch in range(max_epochs): #loop through the max number of iterations
        get all currently misclassified examples
        if no misclassified examples: #Once we classified everything correctly, break out of the loop
                break
        X, Y = randomly select one misclassified example
        w = w + n*(Y)*(X) // n = 1.0, 0.1, 0.01, etc. typically #n = eita = learning_rate
        #Repeat the process until max_epochs, or no missclassified samples left

