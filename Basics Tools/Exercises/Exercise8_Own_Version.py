import numpy as np
import matplotlib.pyplot as plt

def create_spiral():
    #Logic:
    #       radius -> increases from low to high
    #       (don't start at 0, otherwise points will be concentrated at the origin)
    #       angle -> increases from low to high, proportional to the radius
    #       example = [0, 2pi/6, 4pi/6, ..., 10pi/6]  ->  [pi/2, pi/3, pi/4, ..., ]
    #       As usual the circles will be created using:
    #           x = rcos(theta) - a/s (Where r is s)
    #           y = rsin(theta) - t/s (Where r is s)

    radius = np.linspace(1, 10, 100)
    thetas = np.empty((6, 100))       #There are 6 arms in the spiral
    #Generates the values for the spiral arms
    for i in range(6):
        start_angle = np.pi*i/3
        end_angle = start_angle + np.pi/2
        points = np.linspace(start_angle, end_angle, 100)
        thetas[i] = points
    
    #Converts the data into cartesian/spherical coordinates:
    x1 = np.empty((6, 100))
    x2 = np.empty((6, 100))
    for i in range(6):
        x1[i] = radius * np.cos(thetas[i]) #Creates the x values
        x2[i] = radius * np.sin(thetas[i]) #Creates the y values
    
    #Inputs
    X = np.empty((600, 2))
    X[:,0] = x1.flatten()
    X[:,1] = x2.flatten()

    #Add noise:
    X += np.random.randn(600, 2)*0.5

    #Targets
    Y = np.array([0]*100 + [1]*100 + [0]*100 + [1]*100 + [0]*100 + [1]*100)
    return X, Y

X, Y = create_spiral()
plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.show()