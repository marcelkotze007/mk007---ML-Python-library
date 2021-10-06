import numpy as np
import matplotlib.pyplot as plt

def create_donut():
    N = 1000
    Inner_R = 10
    Outer_R = 20
    print(N)
    #Distance from the origin = radius + random normal
    #Angle theta is uniformly distributed between (0, 2pi)

    R1 = np.random.randn(N) + Inner_R
    theta = 2*np.pi*np.random.random(N)
    X_inner = np.concatenate([[R1 * np.cos(theta)], [R1 * np.sin(theta)]]).T #Create concentric circle inner

    R2 = np.random.randn(N) + Outer_R
    theta = 2*np.pi*np.random.random(N)
    X_outer = np.concatenate([[R2 * np.cos(theta)], [R2 * np.sin(theta)]]).T #Create concentric circle outer

    X = np.concatenate([X_inner, X_outer])
    Y = np.array([0]*(N) + [1]*(N))
    return X, Y

X, Y = create_donut()
plt.scatter(X[:, 0], X[:, 1], c = Y)
plt.show()

